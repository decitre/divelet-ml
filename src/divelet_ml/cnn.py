"""The convolutional block of the book's Chapter 8 that later sections use.

Adapted from `d2l/torch.py` in Dive into Deep Learning (https://github.com/d2l-ai/d2l-en, commit
23d7a5aecceee57d1292c56e90cce307f183bb0a), Copyright 2019 Amazon.com, Inc. or its affiliates, MIT-0 license.
"""

from divelet_ml import nn


class ResNeXtBlock(nn.Module):
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False, strides=1):
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3, stride=strides, padding=1, groups=bot_channels // groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1, self.bn2, self.bn3 = nn.LazyBatchNorm2d(), nn.LazyBatchNorm2d(), nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = nn.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return nn.relu(Y + X)
