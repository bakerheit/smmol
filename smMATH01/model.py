"""smMATH01: a tiny GPT that does whole-number arithmetic in its weights.

Three variants with the same size and the same training. They differ only in digit order and in how a
token knows where it is:

    plain     digits in the usual order, a learned embedding for each position in the line (like smLLM_01)
    reversed  every number ones-digit first, with the same position embeddings
    abacus    ones-digit first, and each digit is labeled by its place inside its own number instead
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from problems import END, PAD, VOCAB_SIZE, places

VARIANTS = ("plain", "reversed", "abacus")


@dataclass
class Config:
    variant: str = "abacus"
    vocab: int = VOCAB_SIZE
    ctx: int = 64         # the longest test line (10 digits times 10 digits) is 43 tokens
    places: int = 64      # place labels, with room for the random shift used in training
    d: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.0  # every problem is fresh, so there's nothing to memorize

    @property
    def reverse(self):
        return self.variant != "plain"


class Attention(nn.Module):
    """Each token looks back at earlier tokens and pulls in what's useful."""

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
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # causal: no peeking ahead
        return self.drop(self.out(y.transpose(1, 2).reshape(b, t, d)))


class Block(nn.Module):
    """One layer: attention mixes across positions, then an MLP works on each position."""

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
        x = x + self.attn(self.n1(x))  # residual stream: each layer adds, never replaces
        return x + self.mlp(self.n2(x))


class MathGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        if c.variant not in VARIANTS:
            raise ValueError("variant must be one of: %s" % ", ".join(VARIANTS))
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        if c.variant == "abacus":
            self.place = nn.Embedding(c.places, c.d)
        else:
            self.pos = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.tok.weight  # reading and writing share one token table
        self.apply(self._init)
        # Start the layers that write into the residual stream small, so a deep stack stays stable.
        for name, p in self.named_parameters():
            if name.endswith("attn.out.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * c.layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, place=None, targets=None):
        if self.c.variant == "abacus":
            where = self.place(places(idx) if place is None else place)
        else:
            where = self.pos(torch.arange(idx.shape[1], device=idx.device))
        x = self.tok(idx) + where
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def answer(self, idx, max_new):
        """Greedy decoding: take the most likely token every time, until every line has written "$"."""
        max_new = min(max_new, self.c.ctx - idx.shape[1])
        done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new):
            nxt = self(idx)[0][:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            idx = torch.cat([idx, nxt[:, None]], dim=1)
            done |= nxt == END
            if bool(done.all()):
                break
        return idx
