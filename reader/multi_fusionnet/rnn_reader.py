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
        self.pos_embedding = nn.Embedding(args.pos_vocab_size, 12, padding_idx=0)
        self.ner_embedding = nn.Embedding(args.ner_vocab_size, 8, padding_idx=0)
        args.char_emb = 50
        self.emb_rnn = nn.GRU(args.char_emb,
                              args.char_emb,
                              batch_first=True,
                              bidirectional=True)
        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        self.embedding.weight.requires_grad=False

        self.emb_attn = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        # RNN document encoder

        pdim = args.embedding_dim * 2 + 50 * 2 + 12 + 8
        qdim = args.embedding_dim + 50 * 2 + 12 + 8

        #self.emb_hw = layers.Highway( 2, dim, gate_bias=-2)

        self.enc_rnn_p = layers.StackedBRNN(
            input_size=pdim,
            hidden_size=128,
            num_layers=2,
            dropout_rate=0.2,
            dropout_output=True,
            concat_layers=True,
            rnn_type=nn.GRU,
            padding=True,
        )

        self.enc_rnn_q = layers.StackedBRNN(
            input_size=qdim,
            hidden_size=128,
            num_layers=2,
            dropout_rate=0.2,
            dropout_output=True,
            concat_layers=True,
            rnn_type=nn.GRU,
            padding=True,
        )


        self.enc_rnn_qu = layers.StackedBRNN(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            dropout_rate=0.2,
            dropout_output=True,
            concat_layers=False,
            rnn_type=nn.GRU,
            padding=True,
        )
        full_size = 300 + 100 + 256 + 256
        hidden_size =  256 * 3
        num_level = 3
        self.full_attn = layers.FullAttention(full_size, hidden_size, num_level)

        self.enc_rnn_fusion = layers.StackedBRNN(
            input_size=256 * 5,
            hidden_size=128,
            num_layers=1,
            dropout_rate=0.2,
            dropout_output=True,
            concat_layers=False,
            rnn_type=nn.GRU,
            padding=True,
        )

        full_size = 300 + 100 + 256 * 2 + 256 * 3 + 256
        hidden_size = 256
        num_level = 1
        self.self_full_attn = layers.FullAttention(full_size, hidden_size, num_level)

        self.enc_rnn_final = layers.StackedBRNN(
            input_size=256 * 2,
            hidden_size=128,
            num_layers=1,
            dropout_rate=0.2,
            dropout_output=True,
            concat_layers=False,
            rnn_type=nn.GRU,
            padding=True,
        )


        self.q_agg = layers.LinearSeqAttn(256)

        self.start_attn = layers.BilinearSeqAttn(256)
        self.end_attn = layers.BilinearSeqAttn(256)


    def forward( self, x1, x1_pos, x1_ner, x1_c, x1_mask, x2, x2_pos, x2_ner, x2_c, x2_mask, x1_f ):
        """
        Args:
            x1 = document word indices             [batch * len_d]
            x1_p = document pos indices             [batch * len_d]
            x1_e = document ner indices             [batch * len_d]
            x1_c = document char indices           [batch * len_d * word_len]
            x1_f = document features               [batch * len_d * num_features]
            x1_mask = document padding mask        [batch * len_d]
            x2 = question word indices             [batch * len_q]
            x2_p = question pos indices             [batch * len_q]
            x2_e = question ner indices             [batch * len_q]
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

        # Embed pos and ner
        x1_pos_emb = self.pos_embedding(x1_pos)
        x2_pos_emb = self.pos_embedding(x2_pos)

        x1_ner_emb = self.ner_embedding(x1_ner)
        x2_ner_emb = self.ner_embedding(x2_ner)

        x1_attn_emb = self.emb_attn(x1_emb, x2_emb, x2_mask)

        # concatenate the embedding from char and word
        x1 = torch.cat([x1_c_emb, x1_emb, x1_pos_emb, x1_ner_emb, x1_attn_emb], dim=2)
        x2 = torch.cat([x2_c_emb, x2_emb, x2_pos_emb, x2_ner_emb], dim=2)

        # Drop word
        #x1 = F.dropout2d(x1.unsqueeze(3), p=0.1, training=self.training).squeeze(3)
        #x2 = F.dropout2d(x2.unsqueeze(3), p=0.1, training=self.training).squeeze(3)

        # Drop some dimension in word dim
        x1 = layers.dropout( x1, p=0.2, training=self.training)
        x2 = layers.dropout( x2, p=0.2, training=self.training)

        # preprocess
        x1_h = self.enc_rnn_p(x1, x1_mask)
        x2_h = self.enc_rnn_q(x2, x2_mask)

        # preprocess
        x2_u = self.enc_rnn_qu(x2_h, x2_mask)

        # inter-attention
        x1_HoW = torch.cat([x1_emb, x1_c_emb, x1_h], dim=2)
        x2_HoW = torch.cat([x2_emb, x2_c_emb, x2_h], dim=2)
        x2_value = torch.cat([x2_h, x2_u], dim=2)
        x1_attn = self.full_attn(x1_HoW, x2_HoW, x2_value, x2_mask)

        # preprocess
        x1_v = self.enc_rnn_fusion(torch.cat([x1_h,x1_attn], dim=2), x1_mask)

        # intra-attention
        X1_HoW_self = torch.cat([x1_emb, x1_c_emb, x1_h, x1_attn, x1_v], dim=2)
        x1_attn_self  = self.self_full_attn(X1_HoW_self, X1_HoW_self, x1_v, x1_mask)
        x1_u = self.enc_rnn_final(torch.cat([x1_v, x1_attn_self], dim=2), x1_mask)

        x2_vec = self.q_agg(x2_u,x2_mask)

        start_scores = self.start_attn(x1_u, x2_vec, x1_mask)
        end_scores = self.end_attn(x1_u, x2_vec, x1_mask)

        return start_scores, end_scores
