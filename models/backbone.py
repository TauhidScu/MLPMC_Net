# -*- coding: utf-8 -*-
"""
Backbone module for MLPMC-Net: ResNet-50 with optional dilation and BaseNet wrapper.
"""

import math
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['BaseNet', 'resnet50', 'Bottleneck', 'conv3x3']

MODEL_URLS = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    """3x3 convolution with padding and optional dilation."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride,
        padding=padding, bias=False, dilation=dilation
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None,
                 dilation=(1, 1), batchnorm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        batchnorm = batchnorm or nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = batchnorm(planes)
        self.conv2 = conv3x3(
            planes, planes, stride,
            padding=dilation[1], dilation=dilation[1]
        )
        self.bn2 = batchnorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = batchnorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class DilatedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dilated=True,
                 batchnorm=nn.BatchNorm2d, pooling='max', **kwargs):
        super(DilatedResNet, self).__init__()
        batchnorm = batchnorm or nn.BatchNorm2d
        self.in_channels = 64
        self.batchnorm = batchnorm

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = batchnorm(64)
        self.relu = nn.ReLU(inplace=True)

        if pooling == 'max':
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif pooling == 'conv':
            self.maxpool = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                batchnorm(64),
                nn.ReLU(inplace=True)
            )
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=(1, 1))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=(1, 1))

        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=(1, 2))
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=(2, 4))
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=(1, 1))
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=(1, 1)):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.batchnorm(planes * block.expansion),
            )

        layers = [block(self.in_channels, planes, stride, downsample,
                        dilation=(1, 1) if dilation[0] == 1 else dilation,
                        batchnorm=self.batchnorm)]
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, dilation=(1, 1) if dilation[1] == 1 else dilation,
                                batchnorm=self.batchnorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4


def resnet50(pretrained=False, pretrained_path=None, **kwargs):
    """Constructs a ResNet-50 model with optional pretrained weights."""
    model = DilatedResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if pretrained_path and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=False)
        else:
            try:
                state_dict = model_zoo.load_url(MODEL_URLS['resnet50'])
                # Adjust for single-channel input if conv1 expects 1 channel
                if state_dict['conv1.weight'].shape[1] == 3 and model.conv1.in_channels == 1:
                    state_dict['conv1.weight'] = state_dict['conv1.weight'].mean(dim=1, keepdim=True)
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"[Warning] Could not load online pretrained weights for ResNet50: {e}")
    return model


class BaseNet(nn.Module):
    """
    Base network wrapping the backbone encoder and extracting multi-level features:
    c1, c2, c3, c4.
    """
    def __init__(self, n_class, backbone='resnet50', in_channels=[256, 512, 1024, 2048],
                 dilated=True, batchnorm=nn.BatchNorm2d, pretrained=False,
                 pretrained_path=None, pooling='max', **kwargs):
        super(BaseNet, self).__init__()
        self.n_class = n_class
        self.backbone = backbone
        self.batchnorm = batchnorm or nn.BatchNorm2d

        if backbone == 'resnet50':
            self.pretrain_model = resnet50(
                pretrained=pretrained,
                pretrained_path=pretrained_path,
                dilated=dilated,
                batchnorm=self.batchnorm,
                pooling=pooling,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Only 'resnet50' is supported in MLPMC.")

    def base_forward(self, x):
        """Extract multi-level features from the backbone."""
        x = self.pretrain_model.conv1(x)
        x = self.pretrain_model.bn1(x)
        x = self.pretrain_model.relu(x)
        x = self.pretrain_model.maxpool(x)

        c1 = self.pretrain_model.layer1(x)
        c2 = self.pretrain_model.layer2(c1)
        c3 = self.pretrain_model.layer3(c2)
        c4 = self.pretrain_model.layer4(c3)

        return c1, c2, c3, c4
