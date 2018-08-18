#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""

import ipdb
import torch
import torch.nn as nn
from torch.nn import functional as F
from . import layers
#import layers


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.args = args

        self.char_embedding = nn.Embedding(args.char_vocab_size,
                                           100,
                                           padding_idx=0)
        self.emb_rnn = nn.GRU(100,
                              100,
                              batch_first=True,
                              bidirectional=True)

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        self.embedding.weight.requires_grad=False


        # Highway network for initial encoding
        self.embhighway = layers.Highway(2, 500)



        #Embedding Encoder Layer
        self.q_enc = layers.Encoder(500, 500)
        self.c_enc = layers.Encoder(500, 500)


        self.qcattn = layers.CQattn(500)

        self.compose = nn.Linear(2000,500)
        # model encoder layer
        self.modelenc1 = layers.Encoder(500,500)
        self.modelenc2 = layers.Encoder(500,500)
        self.modelenc3 = layers.Encoder(500,500)
        self.linears = nn.Linear(1000,1, bias=False)
        self.lineare = nn.Linear(1000,1, bias=False)




    def forward(self, x1, x1_c, x1_mask, x2, x2_c, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_c = document char indices           [batch * len_d * word_len]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_c = question char indices           [batch * len_q * word_len]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed char of doc and question
        b, sl, wl = x1_c.size()
        x1_c_emb = self.char_embedding(x1_c.view(b * sl, wl))
        _ , x1_c_emb = self.emb_rnn(x1_c_emb)
        x1_c_emb = torch.cat(list(x1_c_emb), dim=1).view(b, sl, -1)

        b, sl, wl = x2_c.size()
        x2_c_emb = self.char_embedding(x2_c.view(b*sl, wl))
        _, x2_c_emb = self.emb_rnn(x2_c_emb)
        x2_c_emb = torch.cat(list(x2_c_emb), dim=1).view(b, sl, -1)


        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)


        x1_emb = torch.cat([x1_c_emb, x1_emb], dim=2)
        x2_emb = torch.cat([x2_c_emb, x2_emb], dim=2)


        x1_emb = self.embhighway(x1_emb)
        x2_emb = self.embhighway(x2_emb)

        x1enc = self.c_enc(x1, x1_emb, x1_mask)
        x2enc = self.q_enc(x2, x2_emb, x2_mask)
        def pp(na,n):
            print(na,n.size())
        A, B = self.qcattn(x1enc, x2enc, x1_mask, x2_mask)
        fusion = torch.cat([x1enc, A, A * x1enc, B * x1enc], dim=2)
        fusion = self.compose(fusion)
        M1 = self.modelenc1(x1, fusion, x1_mask)
        M2 = self.modelenc2(x1, M1, x1_mask)
       # M3 = self.modelenc3(x1, M2, x1_mask)
        starts = self.linears(torch.cat([M1,M2],dim=2)).squeeze(2)
        ends = self.lineare(torch.cat([M1,M2],dim=2)).squeeze(2)
        if self.training:
            start_scores = F.log_softmax(starts,dim=1)
            end_scores = F.log_softmax(ends, dim=1)
        else:
            start_scores = F.softmax(starts,dim=1)
            end_scores = F.softmax(ends, dim=1)
        return start_scores, end_scores




