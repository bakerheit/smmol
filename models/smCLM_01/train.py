#!/usr/bin/env python3
"""Train the idea reader and the word reader on the same sentences, then ask both what the held-out words mean.

The idea reader never sees a noun as a word: every noun comes in as the ideas it has. Held-out
nouns (car among them) come in with no ideas at all. After each pass over the sentences, the
reader looks at every place a held-out word was used, predicts the ideas that fit there, and
that average becomes the word's ideas.

    python3 train.py                   # 3 seeds, a few minutes on the CPU
    python3 train.py --seeds 1 --epochs 4
"""
import argparse
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F

from model import MASK, UNK, ConceptModel, WordModel
from world import World

HERE = os.path.dirname(os.path.abspath(__file__))

def recorded_args(args):
    """argparse values for results.json, with paths under this project made relative.

    --out defaults to a path under HERE, which is absolute on whoever's machine ran the
    training. Recording it verbatim writes a home directory into a file that gets published
    and tells a reader nothing they need.
    """
    here = str(HERE)
    clean = {}
    for key, value in vars(args).items():
        text = str(value)
        clean[key] = os.path.relpath(text, here) if text.startswith(here) else value
    return clean



def encode(world, corpus, max_len):
    frame_id = {w: i + 3 for i, w in enumerate(world.frames)}
    noun_id = {w: i for i, w in enumerate(world.nouns)}
    frames, nouns, pad = [], [], []
    for toks in corpus:
        f, n, p = [0] * max_len, [-1] * max_len, [True] * max_len
        for j, t in enumerate(toks[:max_len]):
            p[j] = False
            if t in noun_id:
                n[j] = noun_id[t]
            else:
                f[j] = frame_id.get(t, UNK)
        frames.append(f)
        nouns.append(n)
        pad.append(p)
    return torch.tensor(frames), torch.tensor(nouns), torch.tensor(pad)


def positions(nouns, keep):
    """(rows, cols) of every noun whose id passes `keep`."""
    rows, cols = (nouns >= 0).nonzero(as_tuple=True)
    ok = keep[nouns[rows, cols]]
    return rows[ok], cols[ok]


def masked(data, rows, cols, ideas=None):
    """A batch with one noun per sentence hidden behind MASK."""
    frames, nouns, pad = data
    f, n, p = frames[rows].clone(), nouns[rows].clone(), pad[rows]
    i = torch.arange(len(rows))
    target = n[i, cols].clone()
    f[i, cols] = MASK
    n[i, cols] = -1
    is_noun = n >= 0
    x = ideas[n.clamp(min=0)] * is_noun.unsqueeze(-1) if ideas is not None else None
    return f, n, is_noun, p, target, x


@torch.no_grad()
def infer_ideas(model, data, rows, cols, ideas, held, batch):
    """For each held-out word: the average ideas the reader predicts wherever the word was hidden."""
    model.eval()
    sums, counts = torch.zeros_like(ideas), torch.zeros(len(ideas))
    for s in range(0, len(rows), batch):
        r, c = rows[s:s + batch], cols[s:s + batch]
        f, _, is_noun, p, target, x = masked(data, r, c, ideas)
        sums.index_add_(0, target, torch.sigmoid(model(f, x, is_noun, c, p)))
        counts.index_add_(0, target, torch.ones(len(r)))
    model.train()
    new = ideas.clone()
    seen = held & (counts > 0)
    new[seen] = sums[seen] / counts[seen].unsqueeze(1)
    return new


def train_idea_reader(world, data, tags, held, args, seed, say):
    torch.manual_seed(seed)
    model = ConceptModel(len(world.frames), len(world.concepts), args.d, args.layers, args.heads, args.max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    ideas = tags.clone()
    ideas[held] = 0.0  # held-out words start with no ideas at all
    rows, cols = positions(data[1], ~held)
    hrows, hcols = positions(data[1], held)
    for epoch in range(1, args.epochs + 1):
        order, total = torch.randperm(len(rows)), 0.0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            f, _, is_noun, p, target, x = masked(data, rows[b], cols[b], ideas)
            loss = F.binary_cross_entropy_with_logits(model(f, x, is_noun, cols[b], p), tags[target])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        ideas = infer_ideas(model, data, hrows, hcols, ideas, held, args.batch)
        say("  idea reader  epoch %d  loss %.4f  held-out precision %.2f" % (epoch, total / len(rows), score(ideas, tags, held)[0]))
    return model, ideas


def train_word_reader(world, data, args, seed, say):
    torch.manual_seed(seed)
    model = WordModel(len(world.frames), len(world.nouns), args.d, args.layers, args.heads, args.max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rows, cols = positions(data[1], torch.ones(len(world.nouns), dtype=torch.bool))
    for epoch in range(1, args.epochs + 1):
        order, total = torch.randperm(len(rows)), 0.0
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            f, n, is_noun, p, target, _ = masked(data, rows[b], cols[b])
            loss = F.cross_entropy(model(f, n, is_noun, cols[b], p), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        say("  word reader  epoch %d  loss %.4f" % (epoch, total / len(rows)))
    return model


@torch.no_grad()
def word_reader_guesses(model, data, tags, held, batch):
    """Two fair ways to get ideas out of a word model: borrow from its nearest known words, or from
    the known words it would put in the same spots."""
    model.eval()
    emb = F.normalize(model.noun.weight, dim=1)
    known = (~held).nonzero().squeeze(1)
    nearest = torch.zeros_like(tags)
    for w in held.nonzero().squeeze(1):
        top = known[(emb[known] @ emb[w]).topk(5).indices]
        nearest[w] = tags[top].mean(0)
    rows, cols = positions(data[1], held)
    sums, counts = torch.zeros_like(tags), torch.zeros(len(tags))
    for s in range(0, len(rows), batch):
        r, c = rows[s:s + batch], cols[s:s + batch]
        f, n, is_noun, p, target, _ = masked(data, r, c)
        logits = model(f, n, is_noun, c, p)
        logits[:, held] = float("-inf")  # only known words can lend their ideas
        sums.index_add_(0, target, torch.softmax(logits, dim=1) @ tags)
        counts.index_add_(0, target, torch.ones(len(r)))
    swaps = torch.zeros_like(tags)
    seen = counts > 0
    swaps[seen] = sums[seen] / counts[seen].unsqueeze(1)
    return nearest, swaps


def score(guess, tags, held):
    """(precision at k, average precision) over held-out words, where k is how many ideas the word really has."""
    p_at_k, ap = [], []
    for w in held.nonzero().squeeze(1):
        truth = tags[w]
        k = int(truth.sum())
        hits = truth[guess[w].argsort(descending=True)]
        p_at_k.append(hits[:k].mean().item())
        precision = hits.cumsum(0) / torch.arange(1, len(hits) + 1)
        ap.append((precision * hits).sum().item() / k)
    return statistics.mean(p_at_k), statistics.mean(ap)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sentences", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    say = lambda line: print(line, flush=True)  # noqa: E731

    world = World()
    tags = torch.tensor([world.vector(w) for w in world.nouns])
    held = torch.tensor([w in world.held_out for w in world.nouns])
    say("smCLM_01: %d ideas, %d nouns (%d held out), %d templates, %d sentences, %d seed(s)"
        % (len(world.concepts), len(world.nouns), int(held.sum()), len(world.templates), args.sentences, args.seeds))

    methods = ["idea reader", "word reader: nearest words", "word reader: words that fit the same spots", "common ideas (no reading)"]
    runs, keep = [], None
    started = time.time()
    for seed in range(args.seeds):
        say("seed %d" % seed)
        data = encode(world, world.corpus(args.sentences, seed), args.max_len)
        idea_model, ideas = train_idea_reader(world, data, tags, held, args, seed, say)
        word_model = train_word_reader(world, data, args, seed, say)
        nearest, swaps = word_reader_guesses(word_model, data, tags, held, args.batch)
        common = tags[~held].mean(0).expand_as(tags)
        guesses = [ideas, nearest, swaps, common]
        runs.append({m: score(g, tags, held) for m, g in zip(methods, guesses)})
        say("  " + " | ".join("%s %.2f" % (m, runs[-1][m][0]) for m in methods))
        if seed == 0:
            keep = (idea_model, word_model, ideas, swaps)

    idea_model, word_model, ideas, swaps = keep
    torch.save({"config": {k: getattr(args, k) for k in ("d", "layers", "heads", "max_len")},
                "frames": world.frames, "nouns": world.nouns, "concepts": world.concepts, "held_out": world.held_out,
                "ideas": ideas, "idea_state": idea_model.state_dict(), "word_state": word_model.state_dict()},
               os.path.join(args.out, "models.pt"))

    summary = {}
    for m in methods:
        pk = [r[m][0] for r in runs]
        aps = [r[m][1] for r in runs]
        summary[m] = {"precision_at_k": round(statistics.mean(pk), 3), "precision_sd": round(statistics.pstdev(pk), 3),
                      "average_precision": round(statistics.mean(aps), 3), "ap_sd": round(statistics.pstdev(aps), 3)}
    words = {}
    for i, w in enumerate(world.nouns):
        if not held[i]:
            continue
        top = lambda vec: [(world.concepts[j], round(vec[j].item(), 2)) for j in vec.argsort(descending=True)[:8]]  # noqa: E731
        words[w] = {"truth": sorted(world.tags[w]), "idea_reader": top(ideas[i]), "word_reader_swaps": top(swaps[i])}
    result = {"when": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - started), "args": recorded_args(args),
              "summary": summary, "per_seed": runs, "held_out": words}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(result, f, indent=1)

    say("\n| method | precision@k | average precision |")
    say("|---|---|---|")
    for m in methods:
        s = summary[m]
        say("| %s | %.2f ± %.2f | %.2f ± %.2f |" % (m, s["precision_at_k"], s["precision_sd"], s["average_precision"], s["ap_sd"]))
    say("\nseed 0, what the idea reader worked out for words it was never told about (✓ = right):")
    for w, info in words.items():
        k = len(info["truth"])
        say("  %-12s %s" % (w, "  ".join("%s%s %.2f" % ("✓" if c in info["truth"] else "✗", c, s) for c, s in info["idea_reader"][:k])))
    say("\nsaved %s and results.json in %.0fs" % (os.path.join(args.out, "models.pt"), time.time() - started))


if __name__ == "__main__":
    main()
