import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, shortcut_stride=0):
        super().__init__()

        self.shortcut_stride = shortcut_stride
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        if shortcut_stride:
            self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=3, stride=shortcut_stride, padding=1, bias=False)
            self.shortcut = nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=shortcut_stride, padding=0, bias=False)

        else:
            self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(inner_channels)
        self.conv2 = nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1, bias=False)
        


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

        
        return shortcut + residual