#!/usr/bin/env python
# coding: utf-8
import ipdb
import torch
import layers
attn = layers.SelfAttention(256).cuda()
x1 = torch.Tensor(32,400,256).cuda()
x1_mask = torch.ByteTensor(32,400).cuda().fill_(0)

attn(x1, x1, x1_mask)


