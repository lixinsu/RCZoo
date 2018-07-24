#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from rnn_reader import RnnDocReader

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.char_vocab_size = 100
args.vocab_size = 1000
args.embedding_dim = 300

reader = RnnDocReader(args).cuda()
x1 = Variable(torch.LongTensor(10, 20).cuda())
x1.fill_(1)
x2 = Variable(torch.LongTensor(10, 20).cuda())
x2.fill_(1)

x1_mask = torch.ByteTensor(10, 20).cuda()
x1_mask.fill_(0)
x2_mask = torch.ByteTensor(10, 20).cuda()
x2_mask.fill_(0)

x1_c = Variable(torch.LongTensor(10, 20, 15).cuda())
x1_c.fill_(1)
x2_c = Variable(torch.LongTensor(10, 20, 15).cuda())
x2_c.fill_(1)

reader(x1, x1_c, x1_mask, x2, x2_c, x2_mask)

