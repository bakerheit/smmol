"""smALLM_01's brain: the same tiny GPT as smLLM_01, reading gadget-world tokens instead of bytes."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 33
    ctx: int = 225
    d: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.0  # every episode is brand new, so there's nothing to memorize


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.out = nn.Linear(c.d, c.d, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.drop(self.out(y.transpose(1, 2).reshape(b, t, d)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.LayerNorm(c.d)
        self.attn = Attention(c)
        self.n2 = nn.LayerNorm(c.d)
        self.mlp = nn.Sequential(
            nn.Linear(c.d, 4 * c.d, bias=False),
            nn.GELU(),
            nn.Linear(4 * c.d, c.d, bias=False),
            nn.Dropout(c.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TinyGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("attn.out.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * c.layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        t = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(t, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
