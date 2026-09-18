"""Two small readers of the same sentences: one sees nouns only as bundles of ideas, one sees them as words."""
import torch
import torch.nn as nn

PAD, MASK, UNK = 0, 1, 2  # reserved ids; real frame words start at 3


class Encoder(nn.Module):
    def __init__(self, d, layers, heads, max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, pad):
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))
        return self.norm(self.body(x, src_key_padding_mask=pad))


class ConceptModel(nn.Module):
    """The idea reader. A noun comes in only as the ideas it has (a vector of 0..1 per idea), never as a
    word id, so two words with the same ideas look identical to it. It predicts the ideas of a hidden noun."""

    def __init__(self, n_frames, n_concepts, d=128, layers=2, heads=4, max_len=12):
        super().__init__()
        self.frame = nn.Embedding(n_frames + 3, d)
        self.idea = nn.Linear(n_concepts, d)
        self.encoder = Encoder(d, layers, heads, max_len)
        self.head = nn.Linear(d, n_concepts)

    def forward(self, frames, ideas, is_noun, target, pad):
        noun = is_noun.unsqueeze(-1).float()
        x = self.frame(frames) * (1 - noun) + self.idea(ideas) * noun
        h = self.encoder(x, pad)
        return self.head(h[torch.arange(h.shape[0], device=h.device), target])


class WordModel(nn.Module):
    """The word reader, the usual way: every noun is its own learned vector, and it predicts which word is hidden."""

    def __init__(self, n_frames, n_nouns, d=128, layers=2, heads=4, max_len=12):
        super().__init__()
        self.frame = nn.Embedding(n_frames + 3, d)
        self.noun = nn.Embedding(n_nouns, d)
        self.encoder = Encoder(d, layers, heads, max_len)
        self.head = nn.Linear(d, n_nouns)

    def forward(self, frames, nouns, is_noun, target, pad):
        noun = is_noun.unsqueeze(-1).float()
        x = self.frame(frames) * (1 - noun) + self.noun(nouns.clamp(min=0)) * noun
        h = self.encoder(x, pad)
        return self.head(h[torch.arange(h.shape[0], device=h.device), target])
