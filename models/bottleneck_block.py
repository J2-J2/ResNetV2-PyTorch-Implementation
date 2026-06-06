import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    def __init__(self, in_channels, inner_channels, shortcut_stride=0):
        super().__init__()

        out_channels = inner_channels * 4
        self.shortcut_stride = shortcut_stride

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(inner_channels)
        if shortcut_stride:
            self.conv2 = nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=shortcut_stride, padding=1, bias=False)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=shortcut_stride, padding=0, bias=False)
        else:
            self.conv2 = nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(inner_channels)
        self.conv3 = nn.Conv2d(inner_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        residual = self.bn1(x)
        residual = self.relu(residual)
        if self.shortcut_stride: 
            shortcut = self.shortcut(residual)
        else:
            shortcut = x

        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)

        residual = self.bn3(residual)
        residual = self.relu(residual)
        residual = self.conv3(residual)
        
        return shortcut + residual