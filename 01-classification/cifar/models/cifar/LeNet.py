# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 1029-11-12
# usage: lenet for image classification
#
# --------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    lenet architecture

    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, bias=False)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, bias=False)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(400, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

if __name__ == "__main__":
    inputs = torch.rand(2,3,32,32)
    net = LeNet()
    print (net)
    outputs = net(inputs)
    print (outputs.size())