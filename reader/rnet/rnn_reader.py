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
                                           50,
                                           padding_idx=0)
        self.emb_rnn = nn.GRU(50,
                              50,
                              batch_first=True,
                              bidirectional=True)
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        self.embedding.weight.requires_grad=False


        # Input size to RNN: word emb + question emb + manual features
        # RNN document encoder

        doc_input_size = args.embedding_dim + 50 * 2
        self.encode_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=64,
            num_layers=3,
            dropout_rate=0.2,
            dropout_output=True,
            concat_layers=False,
            rnn_type=nn.GRU,
            padding=True,
        )

        # Output sizes of rnn encoders
        outdim = 2 * 64

        self.docattn = layers.DotAttention(outdim, 128)


        self.attn_rnn1 = layers.StackedBRNN(
                    input_size=outdim*2,
                    hidden_size=128,
                    num_layers=1,
                    dropout_rate=0,
                    dropout_output=True,
                    concat_layers=False,
                    rnn_type=nn.GRU,
                    padding=True,
                )

        # self attention
        self.selfattn = layers.DotAttention(128*2, 128)
        self.attn_rnn2 = layers.StackedBRNN(
                    input_size=128*2*2,
                    hidden_size=128,
                    num_layers=1,
                    dropout_rate=0,
                    dropout_output=False,
                    concat_layers=False,
                    rnn_type=nn.GRU,
                    padding=True,
                )

        # Question merging
        self.q_attn = layers.LinearSeqAttn(2 * 64)

        # pointer network
        self.start_ptr = layers.PtrNet(2 * 64 + 128 * 2)
        self.end_ptr = layers.PtrNet(2 * 64 + 128 * 2)

        self.ptr_rnn = nn.GRUCell( 256, 128)


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

        # Encode the question and context
        c = self.encode_rnn(x1_emb, x1_mask)
        q = self.encode_rnn(x2_emb, x2_mask)
        #c = F.dropout(c, p=0.2, training=self.training)
        #q = F.dropout(q, p=0.2, training=self.training)
        # attention
        qc_att = self.docattn(c, q, x2_mask)

        qc_att = self.attn_rnn1(qc_att, x1_mask)

        # self attention
        self_att = self.selfattn(qc_att, qc_att, x1_mask)

        match = self.attn_rnn2(self_att, x1_mask)

        # self attention convert question to q_vector
        q_vec = self.q_attn(q, x2_mask)

        internal, start_scores = self.start_ptr(match, q_vec, x1_mask)
        internal = self.ptr_rnn(internal, q_vec)
        _, end_scores = self.end_ptr(match, internal, x1_mask)

        if self.training:
            start_scores = torch.log(start_scores + 1e-10)
            end_scores = torch.log(end_scores + 1e-10)
        return start_scores, end_scores
