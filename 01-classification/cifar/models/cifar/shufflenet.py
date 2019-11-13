# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 2019-11-13
# usage: shufflenet for image classification
# --------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """
        shufflenet 最重要的特征就是通道的shuffle，这种shuffle是有规则的，按照一定的组进行分组，然后每组中按照相同idx取出
                    相应的通道组成新的group， 最后将成组的channels还原到原来的通道数
        :param x:
        :return:
        """
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        """
        shufflenet的基础模块bottleneck中，1*1分组卷积后使用shuffle进行通道的打乱；
                    同时将输入与输出进行信息整合，在降采样的时候使用cat，非降采样使用residual连接方式。
        :param in_planes:
        :param out_planes:
        :param stride:
        :param groups:
        """
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes//4
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups = g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, padding=1, stride=stride, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.short = nn.Sequential()
        if stride ==2 :
            self.short = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.short(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out+res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, cfg):
        super(ShuffleNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layers(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layers(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layers(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], 10)

    def _make_layers(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i==0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ShuffleNetG2():
    cfg = {
        'out_planes': [200,400,800],
        'num_blocks': [4, 8, 4],
        'groups':2
    }
    return ShuffleNet(cfg)


def ShuffleNetG3():
    cfg = {
        'out_planes': [240, 480, 960],
        'num_blocks': [4, 8, 4],
        'groups':3
    }
    return ShuffleNet(cfg)


if __name__ == "__main__":
    net = ShuffleNetG2()
    inputs = torch.randn(3,3,32,32)
    outputs = net(inputs)
    print (net)
    print (outputs.size())