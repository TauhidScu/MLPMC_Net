# -*- coding: utf-8 -*-
"""
Models package for MLPMC: Multi-Label Pathology Multi-Task Classification and Segmentation.
"""

from .backbone import BaseNet, resnet50
from .attention import (
    ChannelAttention,
    FeatureVarianceAttention,
    Attention_block,
    PSFFM,
    SegFeatureEmbed,
    LabelConditionedSegClsAttention
)
from .mlpmc_net import (
    MLPMCNet,
    ClsHead,
    SegHead,
    get_model_cls_mlpmc,
    get_segmentation_model_cls_mlpmc,
    grad_scale,
    pcgrad_project
)

__all__ = [
    'BaseNet',
    'resnet50',
    'MLPMCNet',
    'ClsHead',
    'SegHead',
    'get_model_cls_mlpmc',
    'get_segmentation_model_cls_mlpmc',
    'grad_scale',
    'pcgrad_project',
    'ChannelAttention',
    'FeatureVarianceAttention',
    'Attention_block',
    'PSFFM',
    'SegFeatureEmbed',
    'LabelConditionedSegClsAttention'
]
