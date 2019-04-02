#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import functional as F
from reader.layers import Highway, StackedBRNN, BiDAFAttention

# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class Reader(nn.Module):
    """
    Examples:
    >>> import torch
    >>> import collections
    >>> ARGS = collections.namedtuple('ARGS',
        'char_vocab_size vocab_size embedding_dim char_emb dropout use_drop')
    >>> args = ARGS(char_vocab_size=1000, vocab_size=10000, embedding_dim=300,
                     char_emb=50, dropout=0.8, use_drop=True)
    >>> x1 = torch.LongTensor(32, 20).cuda().zero_()
    >>> x2 = torch.LongTensor(32, 25).cuda().zero_()
    >>> x1_mask = torch.ByteTensor(32, 20).cuda().zero_()
    >>> x2_mask = torch.ByteTensor(32, 25).cuda().zero_()
    >>> x1_c = torch.LongTensor(32, 20, 15).cuda().zero_()
    >>> x2_c = torch.LongTensor(32, 25, 15).cuda().zero_()
    >>> reader = Reader(args).cuda()
    >>> start_scores, end_scores = reader(x1, x1_c, x1_mask, x2, x2_c,
                                           x2_mask)
    """

    def __init__(self, args):
        super(Reader, self).__init__()

        self.args = args

        self.char_embedding = nn.Embedding(args.char_vocab_size,
                                           args.char_emb,
                                           padding_idx=0)
        self.emb_rnn = nn.GRU(args.char_emb,
                              args.char_emb,
                              batch_first=True,
                              bidirectional=True)

        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        self.embedding.weight.requires_grad=False

        dim = args.embedding_dim + args.char_emb * 2

        self.emb_hw = Highway(2, dim, gate_bias=-2)

        self.enc_rnn = StackedBRNN(
            input_size=dim,
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.2,
            dropout_output=True,
            concat_layers=True,
            rnn_type=nn.GRU,
            padding=True,
        )

        self.qc_attn = BiDAFAttention(64*4)

        self.linear_attn = nn.Linear(64 * 4 * 4, 64*4)

        self.relu_attn = nn.ReLU()

        self.self_rnn = StackedBRNN(
                    input_size=64*4,
                    hidden_size=64,
                    num_layers=2,
                    dropout_rate=0.2,
                    dropout_output=True,
                    concat_layers=True,
                    rnn_type=nn.GRU,
                    padding=True,
                )

        self.cc_attn = BiDAFAttention(64*4, self_attn=True)

        self.linear_self = nn.Linear(64* 4 * 4, 64*4)
        self.relu_self = nn.ReLU()

        self.fusion_rnn1 = StackedBRNN(
                    input_size=64*4,
                    hidden_size=64,
                    num_layers=2,
                    dropout_rate=0.2,
                    dropout_output=True,
                    concat_layers=True,
                    rnn_type=nn.GRU,
                    padding=True,
                )

        self.start = nn.Sequential(
                    nn.Linear(64*4, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )

        self.fusion_rnn2 = StackedBRNN(
                    input_size=64 * 4 * 2,
                    hidden_size=64,
                    num_layers=2,
                    dropout_rate=0.2,
                    dropout_output=True,
                    concat_layers=True,
                    rnn_type=nn.GRU,
                    padding=True,
                )

        self.end = nn.Sequential(
                nn.Linear(64*4, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
                )

    def forward(self, x1, x1_c, x1_mask, x2, x2_c, x2_mask):
        """
        Inputs:
        x1 = document word indices             [b * sl]
        x1_c = document char indices           [b * sl * wl]
        x1_mask = document padding mask        [b * sl]
        x2 = question word indices             [b * sl]
        x2_c = question char indices           [b * sl * wl]
        x2_mask = question padding mask        [b * sl]
        """
        # Embed char of doc and question
        ipdb.set_trace()
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

        x1_pro = self.emb_hw(x1_emb)
        x2_pro = self.emb_hw(x2_emb)

        x1_pro = self.enc_rnn(x1_pro, x1_mask)
        x2_pro = self.enc_rnn(x2_pro, x2_mask)
        attn_A, attn_B = self.qc_attn(x1_pro, x1_mask, x2_pro, x2_mask)
        c_fusion = torch.cat( [x1_pro, attn_A, x1_pro * attn_A, attn_A * attn_B], dim=2)
        c_fusion = self.linear_attn(c_fusion)
        c_fusion = self.relu_attn(c_fusion)

        self_attn_input = self.self_rnn(c_fusion, x1_mask)

        self_attn_A, self_attn_B = self.cc_attn(self_attn_input, x1_mask,
                                                self_attn_input, x1_mask)

        self_fusion = torch.cat(
            [self_attn_input, self_attn_A, self_attn_input * self_attn_A, self_attn_A * self_attn_B],
            dim=2)

        self_fusion = self.relu_attn(self.linear_self(self_fusion))

        context = self_fusion + c_fusion    # 256

        start_context = self.fusion_rnn1(context, x1_mask)

        start_logits = self.start(start_context).squeeze(2)

        start_logits.data.masked_fill_(x1_mask.data, -2e20)

        softmax_start = F.softmax(start_logits, dim=1)

        re_fusion = torch.cat( [context, start_context] ,dim=2)

        g2 = self.fusion_rnn2(re_fusion, x1_mask)

        end_logits = self.end(g2).squeeze(2)

        end_logits.data.masked_fill_(x1_mask.data, -2e20)

        softmax_end = F.softmax(end_logits, dim=1)

        if self.training:
            start_scores = torch.log(softmax_start + 1e-20)
            end_scores = torch.log(softmax_end + 1e-20)
        else:
            start_scores = softmax_start
            end_scores = softmax_end

        return start_scores, end_scores



