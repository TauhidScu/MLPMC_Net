# -*- coding: utf-8 -*-
"""
Loss functions for MLPMC:
- Multi-label Focal Loss with support for missing (-1) annotations
- Compound Dice and Binary Cross Entropy Loss (DC_and_BCE_loss)
- Semi-supervised consistency weighting schedule
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'focal_loss',
    'FocalLoss',
    'DC_and_BCE_loss',
    'MemoryEfficientSoftDiceLoss',
    'consistency_weight'
]


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor = None,
               alpha: float = 0.55, gamma: float = 2.0, reduction: str = 'mean') -> torch.Tensor:
    """
    Multi-label binary focal loss with optional validity mask (for unannotated labels marked as -1).

    Args:
        inputs: Raw logits [B, n_class]
        targets: Binary ground truth [B, n_class] in [0, 1]
        valid_mask: Bool tensor [B, n_class], True where target is valid
        alpha: Class weighting factor
        gamma: Focusing parameter
        reduction: 'mean' | 'sum' | 'none'
    """
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
    p_t = torch.exp(-bce_loss)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t) ** gamma * bce_loss

    if valid_mask is not None:
        loss = loss * valid_mask.float()

    if reduction == 'mean':
        if valid_mask is not None:
            n = valid_mask.float().sum().clamp(min=1.0)
            return loss.sum() / n
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


class FocalLoss(nn.Module):
    """Focal loss module wrapper."""
    def __init__(self, alpha: float = 0.55, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        return focal_loss(inputs, targets, valid_mask=valid_mask,
                          alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)


class MemoryEfficientSoftDiceLoss(nn.Module):
    """
    Memory-efficient soft dice loss for binary segmentation.
    """
    def __init__(self, apply_nonlin=torch.sigmoid, batch_dice: bool = False,
                 smooth: float = 1e-5, do_bg: bool = True, ddp: bool = False):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.do_bg = do_bg
        self.ddp = ddp

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Flatten spatial dimensions: [B, C, H, W] -> [B, C, -1]
        dim = list(range(2, x.ndim))

        if loss_mask is not None:
            x = x * loss_mask
            y = y * loss_mask

        axes = dim if not self.batch_dice else [0] + dim

        tp = (x * y).sum(dim=axes)
        fp = (x * (1 - y)).sum(dim=axes)
        fn = ((1 - x) * y).sum(dim=axes)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth).clamp(min=1e-8)
        dc = dc.mean()
        return 1.0 - dc


class DC_and_BCE_loss(nn.Module):
    """
    Compound Dice and BCE loss matching the exact interface used in MLPMC.
    Returns: [total_loss, dice_loss, bce_loss].
    """
    def __init__(self, bce_kwargs: dict = None, soft_dice_kwargs: dict = None,
                 reduction: str = 'mean', weight_ce: float = 1.0, weight_dice: float = 1.0,
                 use_ignore_label: bool = False, sig: bool = True,
                 dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        bce_kwargs = bce_kwargs or {}
        soft_dice_kwargs = soft_dice_kwargs or {}

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label
        self.reduction = reduction

        bce_kwargs['reduction'] = reduction
        self.dc = dice_class(apply_nonlin=torch.sigmoid if sig else None, **soft_dice_kwargs)
        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.sig = sig

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, focal_weight: torch.Tensor = None):
        if self.use_ignore_label and target.shape[1] > 1:
            mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            mask = None
            target_regions = target

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)

        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)

        if self.reduction == 'mean':
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
            return [result, dc_loss, ce_loss]
        else:
            if focal_weight is not None:
                ce_loss = (ce_loss * focal_weight.detach()).mean()
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
            return [result, dc_loss, ce_loss]


def consistency_weight(current_epoch: int, rampup_epochs: int = 40, max_weight: float = 0.2) -> float:
    """
    Sigmoid ramp-up weighting for feature-perturbation semi-supervised consistency.
    Keeps early noisy pseudo-labels from corrupting the decoder.
    """
    if rampup_epochs <= 0:
        return float(max_weight)
    t = float(np.clip(current_epoch / rampup_epochs, 0.0, 1.0))
    return float(max_weight) * float(np.exp(-5.0 * (1.0 - t) ** 2))
