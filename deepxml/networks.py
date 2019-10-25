#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""

import torch.nn as nn

from deepxml.modules import *


__all__ = ['AttentionRNN', 'FastAttentionRNN']


class Network(nn.Module):
    """

    """
    def __init__(self, emb_size, vocab_size=None, emb_init=None, emb_trainable=True, padding_idx=0, emb_dropout=0.2,
                 **kwargs):
        super(Network, self).__init__()
        self.emb = Embedding(vocab_size, emb_size, emb_init, emb_trainable, padding_idx, emb_dropout)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class AttentionRNN(Network):
    """

    """
    def __init__(self, labels_num, emb_size, hidden_size, layers_num, linear_size, dropout, **kwargs):
        super(AttentionRNN, self).__init__(emb_size, **kwargs)
        self.lstm = LSTMEncoder(emb_size, hidden_size, layers_num, dropout)
        self.attention = MLAttention(labels_num, hidden_size * 2)
        self.linear = MLLinear([hidden_size * 2] + linear_size, 1)

    def forward(self, inputs, **kwargs):
        emb_out, lengths, masks = self.emb(inputs, **kwargs)
        rnn_out = self.lstm(emb_out, lengths)   # N, L, hidden_size * 2
        attn_out = self.attention(rnn_out, masks)      # N, labels_num, hidden_size * 2
        return self.linear(attn_out)


class FastAttentionRNN(Network):
    """

    """
    def __init__(self, labels_num, emb_size, hidden_size, layers_num, linear_size, dropout, parallel_attn, **kwargs):
        super(FastAttentionRNN, self).__init__(emb_size, **kwargs)
        self.lstm = LSTMEncoder(emb_size, hidden_size, layers_num, dropout)
        self.attention = FastMLAttention(labels_num, hidden_size * 2, parallel_attn)
        self.linear = MLLinear([hidden_size * 2] + linear_size, 1)

    def forward(self, inputs, candidates, attn_weights: nn.Module, **kwargs):
        emb_out, lengths, masks = self.emb(inputs, **kwargs)
        rnn_out = self.lstm(emb_out, lengths)   # N, L, hidden_size * 2
        attn_out = self.attention(rnn_out, masks, candidates, attn_weights)     # N, sampled_size, hidden_size * 2
        return self.linear(attn_out)
