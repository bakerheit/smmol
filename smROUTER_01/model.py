"""smROUTER_01's network: reads raw bytes of the assistant's last message and the person's message, and answers
three small questions at once: what kind of message, which tool, and ask first or not."""
import torch
import torch.nn as nn

CLS, SEP, PAD = 256, 257, 258
VOCAB = 259
PREV_BYTES = 64  # the end of the assistant's last message is where its question is


def encode(prev, text, max_len):
    p = list(prev.encode("utf-8"))[-PREV_BYTES:] if prev else []
    t = list(text.encode("utf-8"))
    return ([CLS] + p + [SEP] + t)[:max_len]


def encode_batch(pairs, max_len):
    ids = torch.full((len(pairs), max_len), PAD, dtype=torch.long)
    pad = torch.ones(len(pairs), max_len, dtype=torch.bool)
    for i, (prev, text) in enumerate(pairs):
        row = encode(prev, text, max_len)
        ids[i, :len(row)] = torch.tensor(row)
        pad[i, :len(row)] = False
    return ids, pad


def only_allowed(logits, allowed):
    """Tools that are switched off get no probability at all: to the router they don't exist."""
    return logits.masked_fill(~allowed, -1e9)


class RouterNet(nn.Module):
    def __init__(self, n_intents, n_tools, d=192, layers=4, heads=6, max_len=192, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.intent = nn.Linear(d, n_intents)
        self.tool = nn.Linear(d, n_tools)
        self.ask = nn.Linear(d, 2)

    def forward(self, ids, pad):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        h = self.norm(self.body(x, src_key_padding_mask=pad))[:, 0]  # the CLS position reads the whole message
        return self.intent(h), self.tool(h), self.ask(h)
