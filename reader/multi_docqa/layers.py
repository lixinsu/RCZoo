#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Definitions of model layers/NN modules"""

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int, gate_bias=-2):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        for i in range(self.n):
            self.gate[i].bias.data.fill_(gate_bias)

    def forward(self, x):
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x



class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.linear = nn.Linear(num_layers, 1)
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
           # output = torch.cat([x.unsqueeze(3) for x in outputs[1:]], dim=3)
           # output = self.linear(output).squeeze(3)
            output = torch.cat(outputs[1:], 2)

        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
        #    output = torch.cat([x.unsqueeze(3) for x in outputs[1:]], dim=3)
        #    output = self.linear(output).squeeze(3)
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class SelfAttention(nn.Module):
    """
        self attention for docqa model
        :param dim: hidden size
    """
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(3*dim, 1)

    def forward(self, x1, x2, x2_mask):
        """
        :param x1: b x n x d
        :param x2: b x n x d
        :param x2_mask: b x n
        """
        # bxnxmxd
        similarity = x1.bmm(x2.transpose(1, 2))
        # bxnxm
        x2_mask = x2_mask.unsqueeze(1).expand_as(similarity)
        similarity.data.masked_fill_(x2_mask.data, -2e20)
        diag_mask = 1 - torch.diag(torch.ones(x1.size(1)).cuda()).unsqueeze(0).expand(x1.size(0),-1,-1)
        similarity = similarity * diag_mask
        # b x n x m
        # c -> q
        sim_row = F.softmax(similarity, dim=2)
        attn_a = sim_row.bmm(x2)
        return attn_a


class BiAttention(nn.Module):
    """
        biattention in BiDAF model
        :param dim: hidden size
    """
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(3 * dim, 1)

    def forward(self, x1, x1_mask, x2, x2_mask):
        """
        :param x1: b x n x d
        :param x2: b x m x d
        :param x1_mask: b x n
        :param x2_mask: b x m
        """
        # bxnxmxd
        x1_aug = x1.unsqueeze(2).expand(x1.size(0), x1.size(1), x2.size(1), x1.size(2))
        x2_aug = x2.unsqueeze(1).expand(x1.size(0), x1.size(1), x2.size(1), x2.size(2))
        x_input = torch.cat([x1_aug, x2_aug, x1_aug * x2_aug], dim=3)
        similarity = self.linear(x_input).squeeze(3)
        # bxnxm
        x2_mask = x2_mask.unsqueeze(1).expand_as(similarity)
        similarity.data.masked_fill_(x2_mask.data, -2e20)
        # bxnxm
        # c -> q
        sim_row = F.softmax(similarity, dim=2)
        attn_a = sim_row.bmm(x2)
        # q -> c
        x1_mask = x1_mask.unsqueeze(2).expand_as(similarity)
        similarity.data.masked_fill_(x1_mask.data, -2e20)
        sim_col = F.softmax(similarity, dim=1)
        q2c = sim_col.transpose(1,2).bmm(x1)
        attn_b = sim_row.bmm(q2c)
        return attn_a, attn_b

class DotAttention(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, hidden):
        super(DotAttention, self).__init__()
        self.hidden = hidden
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, hidden)
        self.linear2 = nn.Linear(input_size, hidden)
        self.linear3 = nn.Linear(2*input_size, 2*input_size)

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        x_proj = F.relu(self.linear1(x))
        y_proj = F.relu(self.linear2(y))


        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1)) / (self.hidden ** 0.5)

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        res = torch.cat([x, matched_seq], dim=2)
        res = F.dropout(res, p=0.2, training=self.training)

        # add gate
        gate = F.sigmoid(self.linear3(res))

        return res * gate



class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearAnswer(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size):
        super(BilinearAnswer, self).__init__()
        self.linear1 = nn.Linear(x_size, 256)
        self.linear2 = nn.Linear(y_size, 256)

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = F.tanh(self.linear1(y))
        Wx = F.tanh(self.linear2(x))
        xWy = Wx.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        return xWy


class PtrNet(nn.Module):
    def __init__(self, in_size):
        super(PtrNet, self).__init__()
        self.linear1 = nn.Linear(in_size, 128)
        self.linear2 = nn.Linear(128, 1)


    def forward(self, doc, q_vec, x1_mask):
        """
        :param p: B * N * H
        :param q_vec: B * H
        :param x1_mask: B * N
        :return res: B * H
        :return out: B * N
        """
        out = torch.cat([doc, q_vec.unsqueeze(1).expand(q_vec.size(0), doc.size(1), q_vec.size(1))], dim=2)
        out = F.tanh(self.linear1(out))
        out = self.linear2(out).squeeze(2) # B * N
        out.data.masked_fill_(x1_mask.data, -float('inf'))
        out = F.softmax(out)
        res = out.unsqueeze(1).bmm(doc).squeeze(1) # b*h

        return res, out


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        scores = self.linear2(F.relu(self.linear1(x))).squeeze(2)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        res = alpha.unsqueeze(1).bmm(x).squeeze(1)
        return res


