#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import jittor as jt
from jittor import nn
import math

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = jt.zeros(self.n_channels)
        self.reset_parameters()

    def reset_parameters(self):
        # Jittor 使用 init 方式初始化参数
        nn.init.constant_(self.weight, self.gamma)

    def execute(self, x):
        norm = x.sqr().sum(dim=1, keepdims=True).sqrt() + self.eps
        x = x / norm
        # 扩展权重以匹配输入形状
        weight_expanded = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weight_expanded * x
        return out