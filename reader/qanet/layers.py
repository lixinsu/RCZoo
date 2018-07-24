#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Definitions of model layers/NN modules"""

import math
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int, gate_bias=-2):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        #self.gate.bias.data.fill_(gate_bias)

    def forward(self, x):
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class SeparableConv(nn.Module):
    def __init__(self, nb_dim, nb_out, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(nb_dim, nb_dim, kernel_size, groups=nb_dim,
                                                            padding=kernel_size // 2, bias=True)
        self.conv2 = nn.Conv1d(nb_dim, nb_out, 1, groups=1, padding=0, bias=True)

    def forward(self,x):
        """
        :param x: shape(bsz, seqlen, 500)
        """

        out = self.conv1(x.transpose(1,2))
        out = self.conv2(out)
        return out.transpose(1,2)


class Encoder(nn.Module):
    """
    encoder unit
    """
    def __init__(self, nb_dim, nb_out):
        super().__init__()
        self.posemb = SinusoidalPositionalEmbedding(nb_out, 0, False, init_size=1024)
        self.sconv1 = SeparableConv(nb_dim, nb_out, 7)
        self.sconv2 = SeparableConv(nb_out, nb_out, 7)
        self.sconv3 = SeparableConv(nb_out, nb_out, 7)
        self.sconv4 = SeparableConv(nb_out, nb_out, 7)
        self.attn = MultiheadAttention(nb_out, 10, dropout=0.2)
        self.ffc = nn.Linear(nb_out, nb_out)

    def forward(self,ori_x, x, x_mask):
        out0 = self.posemb(ori_x)
        x = out0 + x
        out = self.sconv1(x)
        out =  out + x
        out1 = self.sconv2(out)
        out1 = out1 + out
        out2 = self.sconv3(out1)
        out2 = out2 + out1
        out3 = self.sconv4(out2)
        out3 = out3 + out2
        out, _ = self.attn(out3.transpose(0,1), out3.transpose(0,1), out3.transpose(0,1), x_mask)
        out = out.transpose(0,1)
        out = out + out3
        out1 = self.ffc(out)
        out1 = out1 + out
        return out



class CQattn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim*3, 1, bias=False)

    def forward(self,C, Q, Cmask, Qmask):
        """
        :param C: bsz x n x dim
        :param Q: bsz x m x dim
        :param Cmask: bsz x n
        :param Qmask: bsz x m
        """
        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        # shape: bsz x n x m x dim
        Cmask = Cmask.type(torch.float).unsqueeze(2)
        Qmask = Qmask.type(torch.float).unsqueeze(1)
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = Qt * Ct
        S = torch.cat([Ct,Qt,CQ], dim=3)
        S = self.linear(S).squeeze(3) # bsz x n x m
        # c -> q
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        # q -> c
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = S1.bmm(Q)
        B = S1.bmm(S2.transpose(1,2).bmm(C))
        return A,B


def mask_logits(target, mask):
    return target * (1-mask) +  mask * (-1e30)


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self._mask = None

        self.in_proj_weight = Parameter(torch.Tensor(3*embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3*embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value,
                key_padding_mask=None,
                need_weights=True):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                # this will allow us to concat it with previous value and get
                # just get the previous value
                k = v = q.new(0)
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = q.contiguous().view(tgt_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.data.masked_fill_(
                key_padding_mask.data.unsqueeze(1).unsqueeze(2),
                -float('inf')
            ) # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2*self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2*self.embed_dim)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        return F.linear(input, weight, bias)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor())

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        # recompute/expand embeddings if needed
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            return self.weights[self.padding_idx + seq_len, :].expand(bsz, 1, -1)

        positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()


    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])
