#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, recall_score, precision_score, matthews_corrcoef, \
    cohen_kappa_score
import random
import math
import pickle
from torch import nn, einsum
import math
import torch
import torch.nn.functional as F

from einops import rearrange


def save_file(file, data):
    with open(file, 'wb') as output:
        pickle.dump(data, output)


def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_badcase(best_preds, file_name='eval'):
    cl_file = "./badcase/" + file_name + ".pkl"  # 保存预测结果和概率
    output = open(cl_file, 'wb')
    pickle.dump(best_preds, output)
    output.close()


def metrics1(trues, preds):
    trues = np.concatenate(trues, -1)
    preds = np.concatenate(preds, 0)
    y_pred = preds.argmax(-1)
    acc = sum(preds.argmax(-1) == trues) / len(trues)
    try:
        auc = roc_auc_score(trues, preds[:, 1])
    except:
        save_badcase(trues, 'trues')
        save_badcase(preds, 'preds')
        save_badcase(y_pred, 'y_pred')
        print(trues)
        print(preds[:, 1])

    mcc = matthews_corrcoef(trues, y_pred)
    f1 = f1_score(trues, y_pred, average='weighted')
    kappa = cohen_kappa_score(trues, y_pred)
    return acc, auc, mcc, f1, kappa


def exists(val):
    return val is not None


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class T5RelativePositionBias(nn.Module):
    def __init__(
            self,
            scale,
            causal=False,
            num_buckets=32,
            max_distance=128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
            relative_position,
            causal=True,
            num_buckets=32,
            max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale


class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


# class LaplacianAttnFn(nn.Module):
#     """ https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared """
#
#     def forward(self, x):
#         mu = math.sqrt(0.5)
#         std = math.sqrt((4 * math.pi) ** -1)
#         return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class GAU(nn.Module):
    def __init__(
            self,
            *,
            dim,
            query_key_dim=128,
            expansion_factor=2.,
            add_residual=True,
            causal=False,
            dropout=0.,
            laplace_attn_fn=False,
            rel_pos_bias=False,
            norm_klass=nn.LayerNorm
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = norm_klass(dim)
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        # self.attn_fn = ReLUSquared() if not laplace_attn_fn else LaplacianAttnFn()
        self.attn_fn = ReLUSquared()
        self.rel_pos_bias = T5RelativePositionBias(scale=dim ** 0.5, causal=causal)

        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(query_key_dim, heads=2)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual

    def forward(
            self,
            x,
            rel_pos_bias=None,
            mask=None
    ):
        seq_len, device = x.shape[-2], x.device

        normed_x = self.norm(x)
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)

        qk = self.to_qk(normed_x)
        q, k = self.offsetscale(qk)

        sim = einsum('b i d, b j d -> b i j', q, k)
        #
        # if exists(self.rel_pos_bias):
        #     sim = sim + self.rel_pos_bias(sim)

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = self.attn_fn(sim / seq_len)
        attn = self.dropout(attn)

        if exists(mask):
            # mask = rearrange(mask, 'b j -> b 1 j')
            attn = attn.masked_fill(mask < 1, 0.)

        if self.causal:
            causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = out * gate

        out = self.to_out(out)

        if self.add_residual:
            out = out + x

        return out
