# -*- coding: utf-8 -*-
"""
Attention modules and feature fusion components for MLPMC-Net:
- ChannelAttention
- FeatureVarianceAttention
- Attention_block (Hierarchical attention)
- PSFFM (Pathology-Specific Feature Fusion Module)
- SegFeatureEmbed (Learnable segmentation logit embedding)
- LabelConditionedSegClsAttention (LC-SCA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'ChannelAttention',
    'FeatureVarianceAttention',
    'Attention_block',
    'Mlp',
    'PSFFM',
    'SegFeatureEmbed',
    'LabelConditionedSegClsAttention',
]


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention module.
    Adaptively recalibrates channel-wise feature responses using avg and max pooling.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        mid_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class FeatureVarianceAttention(nn.Module):
    """
    Feature variance-based spatial attention for deep semantic features.
    Computes channel-wise variance at each spatial location to detect heterogeneous lesion regions.
    """
    def __init__(self, in_channels: int = None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Variance across channel dimension
        x_var = torch.var(x, dim=1, keepdim=True)  # [B, 1, H, W]
        return self.encoder(x_var)


class Attention_block(nn.Module):
    """
    Hierarchical multi-scale attention block connecting gating signals (F_g)
    to lower-level feature representations (F_l).
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Mlp(nn.Module):
    """Linear projection with dropout for classification head."""
    def __init__(self, in_features: int, out_feature: int, drop: float = 0.):
        super().__init__()
        self.fc = nn.Linear(in_features, out_feature)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc(x))


class PSFFM(nn.Module):
    """
    Pathology-Specific Feature Fusion Module (PSFFM).
    Combines channel attention and spatial variance attention on deepest features,
    fusing them with multi-scale refined stages.
    """
    def __init__(self, in_channels: int = 1024, reduction: int = 16):
        super().__init__()

        self.sigmoid = nn.Sigmoid()
        self.refine_conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.refine_conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.refine_conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.refine_conv4 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

        # Branch A: Channel attention
        self.channel_att = ChannelAttention(in_channels, reduction=reduction)

        # Branch B: Variance spatial attention
        self.variance_att = FeatureVarianceAttention(in_channels)

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        size = x1.size()[2:]
        if x2.size()[2:] != size:
            x2 = F.interpolate(x2, size, mode='bilinear', align_corners=True)
        if x3.size()[2:] != size:
            x3 = F.interpolate(x3, size, mode='bilinear', align_corners=True)
        if x4.size()[2:] != size:
            x4 = F.interpolate(x4, size, mode='bilinear', align_corners=True)

        refine1 = self.refine_conv1(x1)
        refine2 = self.refine_conv2(x2)
        refine3 = self.refine_conv3(x3)
        refine4 = self.refine_conv4(x4)

        # Branch A
        x1_ca = x1 * self.channel_att(x1)

        # Branch B
        x1_va = x1 * self.variance_att(x1)

        x1_fused = x1_ca + x1_va

        # Weighted final fusion
        w = torch.sigmoid(self.fusion_weights)
        out = w[0] * x1_fused + w[1] * (
            (self.sigmoid(refine2) * x2) +
            (self.sigmoid(refine3) * x3) +
            (self.sigmoid(refine4) * x4)
        )
        return out


class SegFeatureEmbed(nn.Module):
    """
    Learnable embedding of the segmentation probability maps for the final classification layer.
    Extracts coarse spatial layout (adaptive pooling), foreground area, and prediction uncertainty p*(1-p).
    """
    def __init__(self, out_dim: int = 256, grid: int = 8, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.grid = grid
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.mlp = nn.Sequential(
            nn.Linear(grid * grid + 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, seg_logits: torch.Tensor) -> torch.Tensor:
        # seg_logits: [B, 1, H, W]
        x = seg_logits.float()
        p = torch.sigmoid(x).clamp(1e-4, 1.0 - 1e-4)
        grid = self.pool(p).flatten(1)                                  # [B, g*g]
        area = p.mean(dim=(1, 2, 3)).unsqueeze(1)                       # [B, 1]
        unc = (p * (1.0 - p)).mean(dim=(1, 2, 3)).unsqueeze(1)         # [B, 1]
        return self.mlp(torch.cat([grid, area, unc], dim=1))            # [B, out_dim]


class LabelConditionedSegClsAttention(nn.Module):
    """
    Label-Conditioned Segmentation-Classification Cross-Attention (LC-SCA).
    Each label query attends over pooled segmentation decoder features.
    """
    def __init__(self, n_class: int, seg_channels: int, cls_channels: int,
                 embed_dim: int = 256, n_heads: int = 4, grid: int = 16,
                 dropout: float = 0.1, detach_seg: bool = False,
                 use_cls_context: bool = True, gate_init: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.L = n_class
        self.d = embed_dim
        self.h = n_heads
        self.dh = embed_dim // n_heads
        self.grid = grid
        self.detach_seg = detach_seg
        self.use_cls_context = use_cls_context

        self.label_emb = nn.Embedding(n_class, embed_dim)
        nn.init.normal_(self.label_emb.weight, std=0.02)

        self.seg_proj = nn.Conv2d(seg_channels, embed_dim, 1)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        if use_cls_context:
            self.cls_proj = nn.Sequential(
                nn.Linear(cls_channels, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )

        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(embed_dim, 1)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, seg_feat: torch.Tensor, cls_feat: torch.Tensor):
        if self.detach_seg:
            seg_feat = seg_feat.detach()

        B = seg_feat.size(0)
        # Spatial pooling of seg features
        seg_pooled = F.adaptive_avg_pool2d(seg_feat, (self.grid, self.grid))
        seg_k_v = self.seg_proj(seg_pooled)  # [B, d, g, g]
        N = self.grid * self.grid
        k_v = seg_k_v.flatten(2).transpose(1, 2)  # [B, N, d]

        # Queries
        q = self.label_emb.weight.unsqueeze(0).expand(B, -1, -1)  # [B, L, d]
        if self.use_cls_context:
            cls_pooled = F.adaptive_avg_pool2d(cls_feat, (1, 1)).flatten(1)
            ctx = self.cls_proj(cls_pooled).unsqueeze(1)          # [B, 1, d]
            q = q + ctx

        Q = self.q_proj(q).view(B, self.L, self.h, self.dh).transpose(1, 2)
        K = self.k_proj(k_v).view(B, N, self.h, self.dh).transpose(1, 2)
        V = self.v_proj(k_v).view(B, N, self.h, self.dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dh ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn_drop = self.dropout(attn)

        attended = torch.matmul(attn_drop, V)  # [B, h, L, dh]
        attended = attended.transpose(1, 2).contiguous().view(B, self.L, self.d)
        evidence = self.o_proj(attended)

        logit_delta = self.out_linear(evidence).squeeze(-1) * self.gate
        attn_map = attn.mean(dim=1).view(B, self.L, self.grid, self.grid)

        return logit_delta, attn_map, evidence
