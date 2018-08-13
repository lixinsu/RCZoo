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
        biattention in SLQA model
        :param dim: hidden size
    """
    def __init__(self, dim):
        super().__init__()
        self.linear_x1 = nn.Linear(dim, dim)
        self.linear_x2 = nn.Linear(dim, dim)


    def forward(self, x1, x1_mask, x2, x2_mask):
        """
        Args:
            x1: b x n x d
            x1_mask: b x n
            x2: b x m x d
            x2_mask: b x m
        Returns:
            attn_a: bxnxd
            attn_b: bxmxd
        """
        # bxnxmxd
        x1_proj = F.relu(self.linear_x1(x1))
        x2_proj = F.relu(self.linear_x2(x2))
        similarity = x1_proj.bmm(x2_proj.transpose(1, 2))
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
        # attention vector
        sim_col = F.softmax(similarity, dim=1)
        attn_b = sim_col.transpose(1,2).bmm(x1)  # bxmxd
        return attn_a, attn_b


class BilinearAttention(nn.Module):
    """
    Bilinear attetinon for slqa
    """
    def __init__(self, hdim):
        super(BilinearAttention, self).__init__()
        self.linear_proj = nn.Linear(hdim, hdim)

    def forward(self, x1, x2, x2_mask):
        """Self attention for SLQA

        This is a general bilinear attention,
        with one learnable matrix

        Args:
            x1: Tensor with size bxnxd
            x2: Tensor with size bxmxd
            x2_mask: ByteTensor with size bxm

        Returns:
            attn_vec: bxnxd

        """
        x2_proj = self.linear_proj(x2)  #bxmxd
        similarity = x1.bmm(x2_proj.transpose(1, 2))   #bxnxm
        x2_mask = x2_mask.unsqueeze(1).expand_as(similarity) #bxnxm
        similarity.data.masked_fill_(x2_mask.data, -2e20)  #bxnxm
        sim_row = F.softmax(similarity, dim=2) #bxnxm
        attn_vec = sim_row.bmm(x2) # bxnxm bxmxd -> bxnxd
        return attn_vec


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    """

    def __init__(self, dim):
        super(BilinearSeqAttn, self).__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y)  #bxd
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)  #bxn
        xWy.data.masked_fill_(x_mask.data, -float('inf'))  #bxn
        if self.training:
            alpha = F.log_softmax(xWy, dim=-1)
        else:
            alpha = F.softmax(xWy, dim=-1)
        return alpha


class FusionLayer(nn.Module):
    """
        make a fusion two vectors
    """
    def __init__(self, hdim):
        super(FusionLayer, self).__init__()
        self.linear_fusion = nn.Linear(hdim * 4, hdim)
        self.linear_gate = nn.Linear(hdim * 4, 1)

    def forward(self, x1, x2):
        """
        Args:
            x1: bxnxd
            x2: bxnxd

        Returns:
            ret: bxnxd
        """
        m = self.linear_fusion(torch.cat([x1, x2, x1 * x2, x1 - x2], dim=2))  # bxnxd
        g = F.sigmoid(self.linear_gate(torch.cat([x1, x2, x1 * x2, x1 - x2], dim=2)))   #bxnx1
        ret = g * m + (1 - g)* x2
        return ret


class LinearSeqAttn(nn.Module):
    """self attention to aggregate question semantics

    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear1 = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """self attention for questin

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)

        Returns:
            alpha: batch * len

        """
        scores = self.linear1(x).squeeze(2)
        scores.data.masked_fill_(x_mask.data, 2e-20)
        alpha = F.softmax(scores, dim=1)
        res = alpha.unsqueeze(1).bmm(x).squeeze(1)
        return res

