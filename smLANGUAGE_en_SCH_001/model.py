"""Configurable byte-level GPT used by smLANGUAGE_en_SCH_001."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int = 256
    context_length: int = 128
    width: int = 192
    layers: int = 4
    heads: int = 6
    dropout: float = 0.1
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, value):
        architecture = value.get("architecture", value)
        config = cls(**{key: architecture[key] for key in asdict(cls()).keys() if key in architecture})
        config.validate()
        return config

    @classmethod
    def from_file(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if not 2 <= self.vocab_size <= 65536:
            raise ValueError("vocab_size must be between 2 and 65536")
        if not 8 <= self.context_length <= 8192:
            raise ValueError("context_length must be between 8 and 8192")
        if self.width < 32 or self.width % self.heads:
            raise ValueError("width must be at least 32 and divisible by heads")
        if not 1 <= self.layers <= 64:
            raise ValueError("layers must be between 1 and 64")
        if not 1 <= self.heads <= 64:
            raise ValueError("heads must be between 1 and 64")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be at least 0 and less than 1")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.proj = nn.Linear(config.width, config.width, bias=False)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.attention_dropout = config.dropout

    def forward(self, x):
        batch, steps, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        q, k, v = (
            value.view(batch, steps, self.heads, width // self.heads).transpose(1, 2)
            for value in (q, k, v)
        )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.residual_dropout(self.proj(attended))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("attention.proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / (2 * config.layers) ** 0.5)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, tokens, targets=None):
        steps = tokens.shape[1]
        if steps > self.config.context_length:
            raise ValueError(
                "sequence length %d exceeds context_length %d" % (steps, self.config.context_length)
            )
        positions = torch.arange(steps, device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, new_tokens, temperature=0.7, top_k=40, repetition_penalty=1.0):
        if new_tokens < 0:
            raise ValueError("new_tokens cannot be negative")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be between 0 and vocab_size")
        if not 1.0 <= repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be between 1 and 2")
        for _ in range(new_tokens):
            window = tokens[:, -self.config.context_length :]
            logits, _ = self(window)
            logits = logits[:, -1]
            if repetition_penalty != 1.0:
                for row in range(tokens.shape[0]):
                    seen = torch.unique(window[row])
                    values = logits[row, seen]
                    logits[row, seen] = torch.where(
                        values < 0, values * repetition_penalty, values / repetition_penalty
                    )
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    cutoff = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1, None]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
