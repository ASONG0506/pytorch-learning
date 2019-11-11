# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time:  2019-11-11
# usage: aux file for cifar-pytorch utilization
# --------------------

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init

def get_mean_and_std(dataset):
    """
    compute the mean and std value of the dataset
    :param dataset:
    :return:
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print ("calculating the mean data")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.weight, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time()
def process_bar():
    pass