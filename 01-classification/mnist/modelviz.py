# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 20200910
# usage: 进行pytorch模型的可视化
# packages： pytorch, tensorflow, tensorboard, tensorboardX
# --------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter  # 用于进行可视化
from torchviz import make_dot


class modelViz(nn.Module):
    def __init__(self):
        super(modelViz, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 10, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        return x


if __name__  == "__main__":
    # 首先来搭建一个模型
    modelviz = modelViz()
    # 创建输入
    sampledata = torch.rand(1, 3, 4, 4)
    # 看看输出结果对不对
    out = modelviz(sampledata)
    print(out)  # 测试有输出，网络没有问题

    # 1. 来用tensorflow进行可视化
    with SummaryWriter("./log", comment="sample_model_visualization") as sw:
        sw.add_graph(modelviz, sampledata)

    # 2. 保存成pt文件后进行可视化
    torch.save(modelviz, "./log/modelviz.pt")

    # 3. 使用graphviz进行可视化
    out = modelviz(sampledata)
    g = make_dot(out)
    g.render('modelviz', view=False)  # 这种方式会生成一个pdf文件


