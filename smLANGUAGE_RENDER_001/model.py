"""A small byte-level GPT that turns compact harness state into an English reply.

Only bytes after SEP are scored. Rotary positions let the same model accept different
input lengths, and the generation cache keeps reply writing cheap on the CPU.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


BYTE_VOCAB = 256
SEP, END, PAD = 256, 257, 258
VOCAB_SIZE = 259


@dataclass
class Config:
    vocab: int = VOCAB_SIZE
    max_len: int = 1024
    d: int = 384
    layers: int = 6
    heads: int = 6
    dropout: float = 0.1


def encode(prompt, reply=None):
    ids = list(prompt.encode("utf-8")) + [SEP]
    return ids if reply is None else ids + list(reply.encode("utf-8")) + [END]


def batch(examples):
    rows = [encode(e["prompt"], e["reply"]) for e in examples]
    ids = torch.full((len(rows), max(map(len, rows))), PAD, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
    x, y = ids[:, :-1], ids[:, 1:].clone()
    after_sep = (x == SEP).long().cumsum(1) > 0
    y[~after_sep | (y == PAD)] = -100
    return x, y


def rope(head_dim, length, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(length).float(), inv)
    return freqs.cos(), freqs.sin()


def rotate(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.head_dim = c.heads, c.d // c.heads
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.out = nn.Linear(c.d, c.d, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin, cache=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, self.head_dim).transpose(1, 2) for z in (q, k, v))
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        return self.drop(self.out(y.transpose(1, 2).reshape(b, t, d))), (k, v)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.LayerNorm(c.d)
        self.attn = Attention(c)
        self.n2 = nn.LayerNorm(c.d)
        self.mlp = nn.Sequential(
            nn.Linear(c.d, 4 * c.d, bias=False), nn.GELU(),
            nn.Linear(4 * c.d, c.d, bias=False), nn.Dropout(c.dropout),
        )

    def forward(self, x, cos, sin, cache=None):
        h, cache = self.attn(self.n1(x), cos, sin, cache)
        x = x + h
        return x + self.mlp(self.n2(x)), cache


class LanguageGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        if c.d % c.heads:
            raise ValueError("d must divide evenly by heads")
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.drop = nn.Dropout(c.dropout)
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
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def _run(self, idx, start=0, caches=None):
        t = idx.shape[1]
        if start + t > self.c.max_len:
            raise ValueError("longer than %d bytes" % self.c.max_len)
        x = self.drop(self.tok(idx))
        cos, sin = self.cos[start:start + t], self.sin[start:start + t]
        fresh = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cos, sin, caches[i] if caches else None)
            fresh.append(cache)
        return self.head(self.norm(x)), fresh

    def forward(self, idx, targets=None):
        logits, _ = self._run(idx)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def write_with_confidence(self, prompt, max_new=400):
        """Return (reply, mean chosen-byte probability, weakest chosen-byte probability)."""
        device = self.tok.weight.device
        encoded = encode(prompt)
        if len(encoded) >= self.c.max_len:
            raise ValueError("input is longer than %d bytes" % (self.c.max_len - 1))
        ids = torch.tensor([encoded], dtype=torch.long, device=device)
        logits, caches = self._run(ids)
        position, out, confidence = ids.shape[1], [], []
        for _ in range(min(max_new, self.c.max_len - position)):
            probs = F.softmax(logits[0, -1], dim=-1)
            nxt = int(probs.argmax())
            confidence.append(float(probs[nxt]))
            if nxt == END:
                break
            if nxt >= BYTE_VOCAB:
                break
            out.append(nxt)
            token = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, caches = self._run(token, position, caches)
            position += 1
        text = bytes(out).decode("utf-8", errors="replace")
        return text, (sum(confidence) / len(confidence) if confidence else 0.0), (min(confidence) if confidence else 0.0)

    @torch.no_grad()
    def write(self, prompt, max_new=400):
        return self.write_with_confidence(prompt, max_new)[0]
