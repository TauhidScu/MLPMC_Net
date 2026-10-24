# -*- coding: utf-8 -*-
"""
MLPMC-Net: Multi-Task Classification + Segmentation Model with Classification-Priority Shared Encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BaseNet
from .attention import (
    Attention_block,
    Mlp,
    PSFFM,
    SegFeatureEmbed,
    LabelConditionedSegClsAttention
)

__all__ = [
    'MLPMCNet',
    'ClsHead',
    'SegHead',
    'get_model_cls_mlpmc',
    'get_segmentation_model_cls_mlpmc',
    'grad_scale',
    'pcgrad_project'
]


# ============================================================
# Gradient routing — shared encoder, classification priority
# ============================================================

class _GradScale(torch.autograd.Function):
    """Identity forward; multiplies the backward gradient by `lam`."""
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x

    @staticmethod
    def backward(ctx, g):
        return g * ctx.lam, None


def grad_scale(x: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Share a feature with a secondary task while scaling its backward gradient.
    - lam == 0.0 -> exact detach (secondary task cannot update the encoder)
    - 0 < lam < 1.0 -> shared, but secondary task's update is attenuated
    - lam == 1.0 -> fully joint training
    """
    if lam == 0.0:
        return x.detach()
    if lam == 1.0:
        return x
    return _GradScale.apply(x, lam)


@torch.no_grad()
def pcgrad_project(shared_params, g_cls_list, g_seg_list, eps: float = 1e-12):
    """
    Projects the segmentation gradient onto the orthogonal complement of the
    classification gradient when they conflict, keeping classification as primary.
    """
    gc = torch.cat([g.reshape(-1) for g in g_cls_list])
    gs = torch.cat([g.reshape(-1) for g in g_seg_list])
    dot = (gs * gc).sum()
    if dot < 0:
        gs = gs - (dot / (gc.pow(2).sum() + eps)) * gc
    g_final = gc + gs
    out, idx = [], 0
    for p in shared_params:
        n = p.numel()
        out.append(g_final[idx:idx + n].view_as(p))
        idx += n
    return out


# ============================================================
# Classification Head
# ============================================================

class ClsHead(nn.Module):
    """
    Classification Head:
    Hierarchical Multi-Scale Attention + PSFFM + Global Pooling + MLP Classifier.
    """
    def __init__(
        self,
        n_class: int,
        in_channels: list = [256, 512, 1024, 2048],
        reduction: int = 16,
        mlp_hidden: int = 1024,
        base_ratio: float = 0.08,
        mlp_drop: float = 0.5,
        use_attention: bool = True,
        use_psffm: bool = True,
        seg_embed_dim: int = 256,
        seg_embed_grid: int = 8,
        use_lcsca: bool = False,
        lcsca_seg_dim: int = 128,
        lcsca_dim: int = 256,
        lcsca_heads: int = 4,
        lcsca_grid: int = 16,
        lcsca_detach_seg: bool = False,
    ):
        super().__init__()
        self.feature_dim = in_channels[3]  # 2048
        self.use_attention = use_attention
        self.use_psffm = use_psffm
        self.seg_embed_dim = seg_embed_dim
        self.use_lcsca = use_lcsca

        # Stage 1: Hierarchical Cross-Stage Attention Blocks
        if self.use_attention:
            self.att2 = Attention_block(F_g=in_channels[1], F_l=in_channels[0], F_int=self.feature_dim)
            self.att3 = Attention_block(F_g=in_channels[2], F_l=in_channels[1], F_int=self.feature_dim)
            self.att4 = Attention_block(F_g=in_channels[3], F_l=in_channels[2], F_int=self.feature_dim)

        # Stage 2: PSFFM Multi-Scale Fusion
        if self.use_psffm:
            self.psffm = PSFFM(in_channels=self.feature_dim, reduction=reduction)

        # 1x1 Channel Projections
        self.c_conv1 = nn.Conv2d(in_channels[0], self.feature_dim, 1, bias=False)
        self.c_conv2 = nn.Conv2d(in_channels[1], self.feature_dim, 1, bias=False)
        self.c_conv3 = nn.Conv2d(in_channels[2], self.feature_dim, 1, bias=False)
        self.c_conv4 = nn.Conv2d(in_channels[3], self.feature_dim, 1, bias=False)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # MLP Classifier
        self.mlp = Mlp(
            in_features=self.feature_dim,
            out_feature=mlp_hidden,
            drop=mlp_drop
        )
        self.fc = nn.Linear(mlp_hidden, n_class)

        # Optional LC-SCA cross-attention
        if self.use_lcsca:
            self.lcsca = LabelConditionedSegClsAttention(
                n_class=n_class,
                seg_channels=lcsca_seg_dim,
                cls_channels=self.feature_dim,
                embed_dim=lcsca_dim,
                n_heads=lcsca_heads,
                grid=lcsca_grid,
                detach_seg=lcsca_detach_seg,
            )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        seg_logits: torch.Tensor = None,
        aspp: torch.Tensor = None,
        seg_feat: torch.Tensor = None
    ) -> torch.Tensor:
        # 1. Hierarchical Attention
        x1_resized = F.interpolate(
            x1, size=x2.shape[2:],
            mode='bilinear', align_corners=False
        )
        if self.use_attention:
            x2_att = self.att2(x1_resized, x2)
            x3_att = self.att3(x2_att, x3)
            x4_att = self.att4(x3_att, x4)
        else:
            x2_att, x3_att, x4_att = x2, x3, x4

        # 2. Channel Projection
        x2_proj = self.c_conv2(x2_att)
        x3_proj = self.c_conv3(x3_att)
        x4_proj = self.c_conv4(x4_att)
        x1_resized_proj = self.c_conv1(x1_resized)

        # 3. PSFFM Fusion
        if self.use_psffm:
            fused_f = self.psffm(x4_proj, x3_proj, x2_proj, x1_resized_proj)
        else:
            fused_f = x4_proj

        # 4. Global Average Pooling & Linear Projection
        gap = self.gap(fused_f).flatten(1)
        logits = self.fc(self.mlp(gap))

        # 5. Optional LC-SCA residual
        if self.use_lcsca and seg_feat is not None:
            delta, _, _ = self.lcsca(seg_feat, fused_f)
            logits = logits + delta

        return logits


# ============================================================
# Segmentation Head
# ============================================================

class SegHead(nn.Module):
    """
    Segmentation Head: U-Net style decoder with depthwise-separable conv blocks
    and multi-scale skip connections.
    """
    def __init__(
        self,
        n_class: int,
        in_channels: list = [256, 512, 1024, 2048],
        reduce_dim: int = 128,
        batchnorm=nn.BatchNorm2d,
        has_aspp: bool = False,
        aspp_dim: int = None,
    ):
        super().__init__()
        batchnorm = batchnorm or nn.BatchNorm2d
        self.has_aspp = has_aspp

        if has_aspp:
            self.aspp_combine = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    reduce_dim + aspp_dim, reduce_dim + aspp_dim,
                    3, 1, 1, bias=False, groups=reduce_dim + aspp_dim
                ),
                batchnorm(reduce_dim + aspp_dim),
                nn.Conv2d(reduce_dim + aspp_dim, reduce_dim, 1, bias=False)
            )

        def _dw_block(ch):
            return nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(ch * 2, ch * 2, 3, 1, 1, bias=False, groups=ch * 2),
                batchnorm(ch * 2),
                nn.Conv2d(ch * 2, ch, 1, bias=False)
            )

        self.up_conv3 = _dw_block(reduce_dim)
        self.up_conv2 = _dw_block(reduce_dim)
        self.up_conv1 = _dw_block(reduce_dim)

        self.conv_out = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, 3, 1, 1),
            batchnorm(reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, n_class, 1)
        )

        self.c_conv1 = nn.Conv2d(in_channels[0], reduce_dim, 1, bias=False)
        self.c_conv2 = nn.Conv2d(in_channels[1], reduce_dim, 1, bias=False)
        self.c_conv3 = nn.Conv2d(in_channels[2], reduce_dim, 1, bias=False)
        self.c_conv4 = nn.Conv2d(in_channels[3], reduce_dim, 1, bias=False)

    def forward(self, x1, x2, x3, x4, aspp=None, return_features: bool = False):
        x1_size = x1.size()[2:]

        x1 = self.c_conv1(x1)
        x2 = self.c_conv2(x2)
        x3 = self.c_conv3(x3)
        x4 = self.c_conv4(x4)

        d4 = x4
        if aspp is not None and self.has_aspp:
            d4 = self.aspp_combine(torch.cat((d4, aspp), dim=1))

        d3 = self.up_conv3(torch.cat(
            (x3, F.interpolate(d4, x3.size()[2:], mode='bilinear', align_corners=True)),
            dim=1
        ))
        d2 = self.up_conv2(torch.cat(
            (x2, F.interpolate(d3, x2.size()[2:], mode='bilinear', align_corners=True)),
            dim=1
        ))
        d1 = self.up_conv1(torch.cat(
            (x1, F.interpolate(d2, x1_size, mode='bilinear', align_corners=True)),
            dim=1
        ))

        logits = self.conv_out(d1)
        if return_features:
            return logits, d1
        return logits


# ============================================================
# MLPMC-Net Main Architecture
# ============================================================

class MLPMCNet(BaseNet):
    """
    MLPMC-Net: Joint Multi-Task Classification + Segmentation Network.
    Uses a single shared encoder with depth-decay gradient attenuation.
    """
    def __init__(
        self,
        n_class: int = 1,
        n_class_cls: int = 4,
        backbone: str = 'resnet50',
        batchnorm=nn.BatchNorm2d,
        aspp_rates=None,
        is_train: bool = False,
        test_size=[256, 256],
        reduce_dim: int = 128,
        pooling: str = 'max',
        task_mode: str = 'both',
        base_ratio: float = 0.08,
        reduction: int = 16,
        use_attention: bool = True,
        use_psffm: bool = True,
        lam_seg_enc=(1.0, 1.0, 0.15, 0.0),
        seg_embed_dim: int = 256,
        seg_embed_grid: int = 8,
        use_lcsca: bool = False,
        lcsca_dim: int = 256,
        lcsca_grid: int = 16,
        lcsca_detach_seg: bool = False,
        **kwargs
    ):
        super().__init__(
            n_class, backbone, batchnorm=batchnorm, pooling=pooling, **kwargs
        )
        assert task_mode in ('classification', 'segmentation', 'both'), (
            f"task_mode must be 'classification', 'segmentation', or 'both', got '{task_mode}'"
        )
        self.is_train = is_train
        self.test_size = test_size
        self.task_mode = task_mode
        self.aspp_rate = aspp_rates
        self.block_channels = [256, 512, 1024, 2048]
        self.use_lcsca = use_lcsca

        # Per-stage gradient scale for seg branch
        if isinstance(lam_seg_enc, (int, float)):
            self.lam_seg_enc = (float(lam_seg_enc),) * 4
        else:
            assert len(lam_seg_enc) == 4, 'lam_seg_enc must be float or 4-tuple'
            self.lam_seg_enc = tuple(float(v) for v in lam_seg_enc)

        # Classification Head
        if task_mode in ('classification', 'both'):
            self.chead = ClsHead(
                n_class=n_class_cls,
                in_channels=self.block_channels,
                base_ratio=base_ratio,
                reduction=reduction,
                use_attention=use_attention,
                use_psffm=use_psffm,
                seg_embed_dim=seg_embed_dim,
                seg_embed_grid=seg_embed_grid,
                use_lcsca=use_lcsca,
                lcsca_seg_dim=reduce_dim,
                lcsca_dim=lcsca_dim,
                lcsca_grid=lcsca_grid,
                lcsca_detach_seg=lcsca_detach_seg,
            )

        # Segmentation Head
        if task_mode in ('segmentation', 'both'):
            self.shead = SegHead(
                n_class=n_class,
                in_channels=self.block_channels,
                reduce_dim=reduce_dim,
                batchnorm=batchnorm,
                has_aspp=(aspp_rates is not None),
                aspp_dim=256 if aspp_rates is not None else None,
            )

    def forward(self, x: torch.Tensor, has_f: bool = False, need_fp: bool = False):
        imsize = x.size()[2:] if self.is_train else self.test_size

        # Backbone features
        c1, c2, c3, c4 = self.base_forward(x)

        # Classification only mode
        if self.task_mode == 'classification':
            return self.chead(c1, c2, c3, c4)

        # Segmentation only mode
        if self.task_mode == 'segmentation':
            out_seg = F.interpolate(
                self.shead(c1, c2, c3, c4),
                imsize, mode='bilinear', align_corners=True
            )
            if need_fp:
                c4_fp = F.dropout2d(c4, p=0.5, training=True)
                out_fp = F.interpolate(
                    self.shead(c1, c2, c3, c4_fp),
                    imsize, mode='bilinear', align_corners=True
                )
                return out_seg, out_fp
            return out_seg

        # Joint mode ('both')
        lam = self.lam_seg_enc
        c1_s = grad_scale(c1, lam[0])
        c2_s = grad_scale(c2, lam[1])
        c3_s = grad_scale(c3, lam[2])
        c4_s = grad_scale(c4, lam[3])

        if self.use_lcsca:
            seg_logits, d1 = self.shead(c1_s, c2_s, c3_s, c4_s, return_features=True)
        else:
            seg_logits, d1 = self.shead(c1_s, c2_s, c3_s, c4_s), None

        out_cls = self.chead(c1, c2, c3, c4, seg_logits=seg_logits, seg_feat=d1)
        out_seg = F.interpolate(
            seg_logits, imsize, mode='bilinear', align_corners=True
        )

        if need_fp:
            c4_s_fp = F.dropout2d(c4_s, p=0.5, training=True)
            out_fp = F.interpolate(
                self.shead(c1_s, c2_s, c3_s, c4_s_fp),
                imsize, mode='bilinear', align_corners=True
            )
            return out_cls, out_seg, out_fp

        return out_cls, out_seg


# ============================================================
# Factory Functions
# ============================================================

def get_model_cls_mlpmc(
    model_name: str = 'mlpmc_net',
    backbone: str = 'resnet50',
    task_mode: str = 'both',
    n_class_cls: int = 4,
    base_ratio: float = 0.08,
    reduction: int = 16,
    use_attention: bool = True,
    use_psffm: bool = True,
    lam_seg_enc=(1.0, 1.0, 0.15, 0.0),
    seg_embed_dim: int = 256,
    seg_embed_grid: int = 8,
    use_lcsca: bool = False,
    lcsca_dim: int = 256,
    lcsca_heads: int = 4,
    lcsca_grid: int = 16,
    lcsca_detach_seg: bool = True,
    **kwargs
) -> MLPMCNet:
    """
    Factory function for instantiating the MLPMC-Net model.
    """
    return MLPMCNet(
        n_class=1,
        n_class_cls=n_class_cls,
        backbone=backbone,
        task_mode=task_mode,
        base_ratio=base_ratio,
        reduction=reduction,
        use_attention=use_attention,
        use_psffm=use_psffm,
        lam_seg_enc=lam_seg_enc,
        seg_embed_dim=seg_embed_dim,
        seg_embed_grid=seg_embed_grid,
        use_lcsca=use_lcsca,
        lcsca_dim=lcsca_dim,
        lcsca_heads=lcsca_heads,
        lcsca_grid=lcsca_grid,
        lcsca_detach_seg=lcsca_detach_seg,
        **kwargs
    )


# Backward-compatible alias
get_segmentation_model_cls_mlpmc = get_model_cls_mlpmc
