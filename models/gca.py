# -*- coding: utf-8 -*-
"""Grouped Cross-Attention (GCA) module for BiMamba-ESCN."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCA(nn.Module):

    def __init__(self, feature_dim: int = 5, dropout: float = 0.0):
        super().__init__()
        self.norm_local = nn.LayerNorm(feature_dim)
        self.norm_global = nn.LayerNorm(feature_dim)
        self.linear_local = nn.Linear(feature_dim, feature_dim)
        self.linear_global = nn.Linear(feature_dim, feature_dim)
        self.linear_final = nn.Linear(feature_dim, feature_dim)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def cross_attention_channelwise(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        
        d = x1.shape[-1]
        scores = torch.matmul(x1, x2.transpose(-1, -2)) / math.sqrt(d)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, x2)

    def bidirectional(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        return self.cross_attention_channelwise(xa, xb) + self.cross_attention_channelwise(xb, xa)

    def forward(self, psd: torch.Tensor, de: torch.Tensor, plv: torch.Tensor, wpli: torch.Tensor) -> torch.Tensor:
        local = self.bidirectional(psd, de)
        global_ = self.bidirectional(plv, wpli)

        local = self.linear_local(self.norm_local(local))
        global_ = self.linear_global(self.norm_global(global_))

        cross = self.bidirectional(local, global_)
        final = self.linear_final(cross)
        return final + local + global_
