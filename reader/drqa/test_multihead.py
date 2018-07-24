#!/usr/bin/env python
# coding: utf-8

import torch
from torch.autograd import Variable
from layers import MultiheadAttention


mm = MultiheadAttention(8, 4)
x = Variable(torch.Tensor(10,2,8))

out, _ = mm(x,x,x,need_weights=False)
print(out.size())
