"""Lot-context head: refine a wafer's prediction using its lot neighbours.

Inputs are frozen CNN embeddings, never labels. The target wafer and up to K
other wafers of the same lot are tokens in a small transformer. Each token
carries the slot offset to the target. The output is a residual added to the
frozen CNN logits, initialised to zero so training starts from the baseline.
"""

import torch
import torch.nn as nn

MAX_OFF = 24  # slot offsets are clipped to +-24, index 24 means "the target itself"
PAD_OFF = 2 * MAX_OFF + 1


class LotContextHead(nn.Module):
    def __init__(self, emb_dim=256, d=64, layers=2, heads=4, drop=0.2, n_cls=9):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, d))
        self.off = nn.Embedding(PAD_OFF + 1, d)
        self.role = nn.Embedding(2, d)  # 1 = target, 0 = neighbour
        layer = nn.TransformerEncoderLayer(
            d, heads, 4 * d, drop, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_cls))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, enc_logits, tgt, nb, nb_off, nb_pad):
        """tgt (B,E), nb (B,K,E), nb_off (B,K) long, nb_pad (B,K) bool True = padding."""
        t = self.proj(tgt) + self.role.weight[1] + self.off.weight[MAX_OFF]
        n = self.proj(nb) + self.role.weight[0] + self.off(nb_off)
        x = torch.cat([t[:, None], n], 1)
        pad = torch.cat([torch.zeros_like(nb_pad[:, :1]), nb_pad], 1)
        h = self.enc(x, src_key_padding_mask=pad)[:, 0]
        return enc_logits + self.out(h)
