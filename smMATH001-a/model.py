"""smMATH001-a: a tiny GPT that works out whole expressions by writing its arithmetic out step by step.

It reads a question from work.py (numbers scaled by 100, ones digit first) and writes the worked steps. Two
things tell a token where it is:

- rotary positions inside attention, so it can find "the line before this one" wherever it is
- a place label on every digit (its place inside its own number), shifted by a random amount in training,
  so the fifth digit of one number lines up with the fifth digit of another

Writing uses a key/value cache, so a long solution costs one short step per token.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = "0123456789+-*/()=:;>"
SEP, END, PAD = len(VOCAB), len(VOCAB) + 1, len(VOCAB) + 2
VOCAB_SIZE = PAD + 1


@dataclass
class Config:
    vocab: int = VOCAB_SIZE
    places: int = 64
    max_len: int = 2048
    d: int = 256
    layers: int = 6
    heads: int = 8


def encode(question, work=None):
    ids = [VOCAB.index(ch) for ch in question] + [SEP]
    return ids if work is None else ids + [VOCAB.index(ch) for ch in work] + [END]


def places(ids):
    """Each digit's place inside its own number, counting from 1 (0 for anything that isn't a digit)."""
    is_digit = ids < 10
    at = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    last_break = torch.where(is_digit, torch.full_like(at, -1), at).cummax(dim=1).values
    return torch.where(is_digit, at - last_break, torch.zeros_like(at))


def batch(examples, rng=None, offset_max=0):
    """(inputs, targets, places). Only the work is scored. In training, every place label in a line shifts together."""
    rows = [encode(e["question"], e["work"]) for e in examples]
    ids = torch.full((len(rows), max(map(len, rows))), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
    x, y = ids[:, :-1], ids[:, 1:].clone()
    after_sep = (x == SEP).long().cumsum(1) > 0
    y[~after_sep | (y == PAD)] = -100
    where = places(x)
    if offset_max and rng is not None:
        shift = torch.tensor([[rng.randint(0, offset_max)] for _ in rows])
        where = torch.where(where > 0, where + shift, where)
    return x, y, where


def rope(head_dim, length, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(length).float(), inv)
    return freqs.cos(), freqs.sin()


def rotate(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    """Each token looks back at earlier tokens, with rotary positions so distances are what it sees."""

    def __init__(self, c):
        super().__init__()
        self.heads, self.head_dim = c.heads, c.d // c.heads
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.out = nn.Linear(c.d, c.d, bias=False)

    def forward(self, x, cos, sin, cache=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, self.head_dim).transpose(1, 2) for z in (q, k, v))
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)  # one new token may see everything cached
        return self.out(y.transpose(1, 2).reshape(b, t, d)), (k, v)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.LayerNorm(c.d)
        self.attn = Attention(c)
        self.n2 = nn.LayerNorm(c.d)
        self.mlp = nn.Sequential(nn.Linear(c.d, 4 * c.d, bias=False), nn.GELU(), nn.Linear(4 * c.d, c.d, bias=False))

    def forward(self, x, cos, sin, cache=None):
        h, cache = self.attn(self.n1(x), cos, sin, cache)
        x = x + h
        return x + self.mlp(self.n2(x)), cache


class MathGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.place = nn.Embedding(c.places, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.tok.weight
        cos, sin = rope(c.d // c.heads, c.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
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

    def _run(self, idx, where, start=0, caches=None):
        t = idx.shape[1]
        if start + t > self.c.max_len:
            raise ValueError("longer than %d tokens" % self.c.max_len)
        x = self.tok(idx) + self.place(where.clamp(max=self.c.places - 1))
        cos, sin = self.cos[start:start + t], self.sin[start:start + t]
        new = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cos, sin, caches[i] if caches else None)
            new.append(cache)
        return self.head(self.norm(x)), new

    def forward(self, idx, where=None, targets=None):
        logits, _ = self._run(idx, places(idx) if where is None else where)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def work(self, question, max_new=None):
        """Greedy decoding with a cache: the worked solution it writes for one question, as text."""
        device = self.tok.weight.device
        ids = torch.tensor([encode(question)], device=device)
        logits, caches = self._run(ids, places(ids))
        position, run, out = ids.shape[1], 0, []
        for _ in range(min(max_new or self.c.max_len, self.c.max_len - position)):
            nxt = int(logits[0, -1].argmax())
            if nxt >= SEP:
                break
            out.append(VOCAB[nxt])
            run = run + 1 if nxt < 10 else 0
            logits, caches = self._run(torch.tensor([[nxt]], device=device), torch.tensor([[run]], device=device), position, caches)
            position += 1
        return "".join(out)
