import torch
import torch.nn as nn
from .basic_block import BasicBlock
from .bottleneck_block import BottleNeck


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, inner_c_list, n_class):
        super().__init__()

        n_channels = 64

        self.conv1 = nn.Conv2d(3, n_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        for i in range(len(num_blocks)):
            if i == 0 and block == BottleNeck: shortcut_stride = 1
            elif i == 0 and block == BasicBlock: shortcut_stride=0
            else: shortcut_stride = 2
            stage, n_channels = self._make_layer(block, num_blocks[i], n_channels, inner_c_list[i], shortcut_stride)
            self.stages.append(stage)
        self.out_bn = nn.BatchNorm2d(n_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_fc = nn.Linear(n_channels, n_class)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        for m in self.modules():
            if isinstance(m, BottleNeck):
                nn.init.zeros_(m.bn3.weight)
            elif isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

                
    def _make_layer(self, block, n_blocks, in_channels, inner_channels, shortcut_stride):
        layers = []
        layers.append(block(in_channels, inner_channels, shortcut_stride))

        if block == BottleNeck: out_channels = inner_channels * 4
        else: out_channels = inner_channels

        for i in range(n_blocks-1):
            layers.append(block(out_channels, inner_channels, False))

        return nn.Sequential(*layers), out_channels
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for stage in self.stages:
            x = stage(x)
        x = self.out_bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.out_fc(x)
        return x

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], inner_c_list=[64, 128, 256, 512], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], inner_c_list=[64, 128, 256, 512], **kwargs)

def resnet50(**kwargs):
    return ResNet(BottleNeck, [3, 4, 6, 3], inner_c_list=[64, 128, 256, 512], **kwargs)

def resnet101(**kwargs):
    return ResNet(BottleNeck, [3, 4, 23, 3], inner_c_list=[64, 128, 256, 512], **kwargs)

def resnet152(**kwargs):
    return ResNet(BottleNeck, [3, 8, 36, 3], inner_c_list=[64, 128, 256, 512], **kwargs)