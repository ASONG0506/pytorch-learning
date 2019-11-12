# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 2019-11-12
# usage: mobilenetv2 for image classification
# reference: https://blog.csdn.net/u011995719/article/details/79135818
# note : some codes are imitated from https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py
# --------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """
    mobilenetv2 basic module
    相比较mobilenetv1,增加了residual block的残差结构，所以需要注意的是不同的通道数的特征对齐与不同的宽高维度的特征对齐的问题：
    对于特征的宽高湿度变化的情况，也就是stride!=1的情况， 直接不进行residual连接了
    对于不同通道数的特征对齐问题：需要将residual结构使用1*1卷积进行通道的对齐
    此外，在block结构中首先需要使用1*1结构进行特征的通道数的增加，这个通过一个expansion参数进行实现
    """
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride = 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride = 1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    """
    mobilenetv2 architecture
    """
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1),
           ]
    def __init__(self, num_classes = 10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes = 32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)


    def _make_layers(self, in_planes):
        layers = []
        for expansion , out_planes, num_blocks, stride in self.cfg:
            strides = [stride] +[1]* ( num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    net = MobileNetV2()
    inputs = torch.randn(3,3,32,32)
    outputs = net(inputs)
    print (net)
    print (outputs.size())