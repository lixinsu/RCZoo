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
        args.char_emb = 50

        self.char_embedding = nn.Embedding(args.char_vocab_size,
                                           50,
                                           padding_idx=0)
        self.emb_rnn = nn.GRU(args.char_emb,
                              args.char_emb,
                              batch_first=True,
                              bidirectional=True)
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        self.embedding.weight.requires_grad=False

        # Input size to RNN: word emb + question emb + manual features
        # RNN document encoder

        dim = args.embedding_dim + args.char_emb * 2

        self.emb_hw = layers.Highway( 2, dim, gate_bias=-2)

        self.enc_rnn = layers.StackedBRNN(
            input_size=dim,
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.2,
            dropout_output=True,
            concat_layers=True,
            rnn_type=nn.GRU,
            padding=True,
        )

        self.qc_attn = layers.BiAttention(64*4)

        self.linear_attn = nn.Linear(64*4*4, 64*4)

        self.relu_attn = nn.ReLU()

        self.cq_rnn = layers.StackedBRNN(
                    input_size=64*4,
                    hidden_size=128,
                    num_layers=1,
                    dropout_rate=0.2,
                    dropout_output=True,
                    concat_layers=False,
                    rnn_type=nn.GRU,
                    padding=True,
                )

        self.cc_attn = layers.SelfAttention(64*4)

        self.linear_self = nn.Linear(64*4, 64*4)
        self.relu_self = nn.ReLU()

        self.fusion_rnn1 = layers.StackedBRNN(
                    input_size=64*4,
                    hidden_size=128,
                    num_layers=2,
                    dropout_rate=0.2,
                    dropout_output=True,
                    concat_layers=False,
                    rnn_type=nn.GRU,
                    padding=True,
                )
        self.start = nn.Sequential(
                    nn.Linear(64*4+128*2, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )

        self.fusion_rnn2 = layers.StackedBRNN(
                    input_size=128*2*3 + 64*4,
                    hidden_size=128,
                    num_layers=1,
                    dropout_rate=0.2,
                    dropout_output=True,
                    concat_layers=False,
                    rnn_type=nn.GRU,
                    padding=True,
                )

        self.end = nn.Sequential(
                nn.Linear(128*2+64*4, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
                )

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


        # concatenate the embedding from char and word
        x1_emb = torch.cat([x1_c_emb, x1_emb], dim=2)
        x2_emb = torch.cat([x2_c_emb, x2_emb], dim=2)

        x1_emb = F.dropout2d(x1_emb.unsqueeze(3), p=0.1, training=self.training).squeeze(3)
        x2_emb = F.dropout2d(x2_emb.unsqueeze(3), p=0.1, training=self.training).squeeze(3)

        # bxnx400
        x1_pro = self.emb_hw(x1_emb)
        x2_pro = self.emb_hw(x2_emb)

        # bxnx(64*4)  bxmx(64*4)
        x1_pro = self.enc_rnn(x1_pro, x1_mask)
        x2_pro = self.enc_rnn(x2_pro, x2_mask)

        # bxnx(64*4) bxnx(64*4)
        attn_A, attn_B = self.qc_attn(x1_pro, x1_mask, x2_pro, x2_mask)

        # bxnx(64*4*4)
        c_fusion = torch.cat( [ x1_pro, attn_A, x1_pro * attn_A, x1_pro * attn_B ], dim=2)

        c_fusion = self.linear_attn(c_fusion)

        c_fusion = self.relu_attn(c_fusion)

        c_fusion_enc = self.cq_rnn( c_fusion, x1_mask)

        # self attention in residual net
        self_attn = self.cc_attn(c_fusion_enc, c_fusion_enc, x1_mask)

        self_attn = self.linear_self(self_attn)

        self_attn = self.relu_self(self_attn)

        c_fusion = c_fusion + self_attn


        # bxnx(128*2)
        g1 = self.fusion_rnn1(c_fusion, x1_mask)

        # input: bxnx(64*4+128*2)
        start_logits = self.start( torch.cat([c_fusion, g1], dim=2) ).squeeze(2)

        start_logits.data.masked_fill_(x1_mask.data, -2e20)

        # bxn
        softmax_start = F.softmax(start_logits, dim=1)


        # bx(128*2)
        a1 = softmax_start.unsqueeze(1).bmm(g1).expand(-1, g1.size(1) ,-1)

        # [p0, g1, a1i, g1 * a1i]  bxnx(128*2*3 + 64*4*4)
        re_fusion = torch.cat( [c_fusion, g1, a1, g1 * a1] ,dim=2)

        g2 = self.fusion_rnn2(re_fusion, x1_mask)

        # input : bxnx(128*2+64*4*4)
        end_logits = self.end(torch.cat([g2, c_fusion], dim=2)).squeeze(2)

        end_logits.data.masked_fill_(x1_mask.data, -2e20)

        softmax_end = F.softmax(end_logits, dim=1)

        if self.training:
            start_scores = torch.log(softmax_start + 1e-20)
            end_scores = torch.log(softmax_end + 1e-20)
        else:
            start_scores = softmax_start
            end_scores = softmax_end

        return start_scores, end_scores



