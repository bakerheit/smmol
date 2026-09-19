"""A small byte-level causal language model for English pretraining."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab: int = 256
    ctx: int = 256
    d: int = 384
    layers: int = 6
    heads: int = 6
    dropout: float = 0.1

    def validate(self):
        if self.d % self.heads:
            raise ValueError("d must be divisible by heads")
        if self.ctx < 2:
            raise ValueError("ctx must be at least 2")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.d, 3 * config.d, bias=False)
        self.proj = nn.Linear(config.d, config.d, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = config.dropout

    def forward(self, x):
        batch, steps, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        q, k, v = (
            value.view(batch, steps, self.heads, width // self.heads).transpose(1, 2)
            for value in (q, k, v)
        )
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d)
        self.mlp = nn.Sequential(
            nn.Linear(config.d, 4 * config.d, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.d, config.d, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab, config.d)
        self.position_embedding = nn.Embedding(config.ctx, config.d)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.d)
        self.output = nn.Linear(config.d, config.vocab, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / (2 * config.layers) ** 0.5)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, tokens, targets=None):
        steps = tokens.shape[1]
        if steps > self.config.ctx:
            raise ValueError(f"sequence length {steps} exceeds context {self.config.ctx}")
        positions = torch.arange(steps, device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.output(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, new_tokens, temperature=0.8, top_k=40):
        for _ in range(new_tokens):
            window = tokens[:, -self.config.ctx :]
            logits, _ = self(window)
            logits = logits[:, -1] / max(temperature, 1e-6)
            if top_k:
                cutoff = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
