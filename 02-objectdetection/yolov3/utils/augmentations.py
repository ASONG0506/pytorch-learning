# -*-coding:utf-8 -*-
# --------------------
# author: cjs
# time: 2019-11-12
# usage: 数据增强
# --------------------
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def horisontal_flip(images, targets):
    """
    水平翻转
    :param images:
    :param targets:
    :return:
    """
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets
