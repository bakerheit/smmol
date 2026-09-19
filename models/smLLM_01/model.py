"""smLLM_01: a tiny byte-level GPT, small enough to read in one sitting."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 256     # one token per byte, so there's no tokenizer to build
    ctx: int = 256       # how many bytes it can see at once
    d: int = 384         # numbers in each token's vector
    layers: int = 6
    heads: int = 6
    dropout: float = 0.2


class Attention(nn.Module):
    """Each byte looks back at earlier bytes and pulls in what's useful."""

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
        self.head.weight = self.tok.weight  # reading and writing share one byte table
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

    def forward(self, idx, targets=None):
        t = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(t, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    @torch.no_grad()
    def generate(self, idx, new_tokens, temperature=0.8, top_k=50):
        """The whole trick: guess the next byte, append it, repeat."""
        for _ in range(new_tokens):
            logits, _ = self(idx[:, -self.c.ctx:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
