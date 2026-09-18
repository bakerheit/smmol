"""smMATH_LANGUAGE_001: a tiny byte-level GPT that reads a message and writes out the arithmetic in it.

    what is the volume of 322234ft x 21323ft x 212231ft?  <SEP>  322234*21323*212231|cubic feet|volume  <END>

The message comes first as plain bytes, then the model writes the problems. Only what it writes is scored.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SEP, END, PAD = 256, 257, 258
MAX_TEXT, MAX_TARGET = 160, 100


@dataclass
class Config:
    vocab: int = PAD + 1
    ctx: int = MAX_TEXT + MAX_TARGET + 2
    d: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.0


def encode(text, target=None):
    ids = list(text.encode("utf-8")[:MAX_TEXT]) + [SEP]
    return ids if target is None else ids + list(target.encode("utf-8")[:MAX_TARGET]) + [END]


def batch(examples):
    """(inputs, targets) with every token before and including <SEP> set to -100, so only the written problems count."""
    rows = [encode(e["text"], e["target"]) for e in examples]
    ids = torch.full((len(rows), max(map(len, rows))), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
    x, y = ids[:, :-1], ids[:, 1:].clone()
    after_sep = (x == SEP).long().cumsum(1) > 0
    y[~after_sep | (y == PAD)] = -100
    return x, y


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


class ReaderGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
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
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def read_with_confidence(self, text):
        """(what it wrote, how sure it was on average, how sure it was at its shakiest character).

        Each character it writes has a probability. The average says how sure the whole answer is, and the
        lowest one catches an answer that was confident everywhere except the digit it got wrong. Ending the
        answer counts as a choice too, so the <END> token's probability is included.
        """
        ids = torch.tensor([encode(text)], device=self.tok.weight.device)
        out, chances = [], []
        for _ in range(MAX_TARGET + 1):
            logits = self(ids)[0][0, -1]
            nxt = int(logits.argmax())
            chances.append(float(torch.softmax(logits, dim=-1)[nxt]))
            if nxt >= SEP:  # <END>, or anything that isn't a byte
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        sure = float(torch.tensor(chances).log().mean().exp()) if chances else 0.0
        return bytes(out).decode("utf-8", errors="replace"), round(sure, 4), round(min(chances), 4) if chances else 0.0

    def read(self, text):
        """Greedy decoding: the target it writes for one message, as text."""
        return self.read_with_confidence(text)[0]
