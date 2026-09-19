"""How well does smALLM_01 understand a brand-new world after k experiments?

For each world it has poked at k times, it's asked: "if you flip each gadget now, what will
every gadget read?" Guesses are scored against the real wiring.
"""
import argparse
import os

import torch
import torch.nn.functional as F

from model import Config, TinyGPT
from world import GADGETS, Worlds, action_tokens, random_flip, reading_tokens, start

HERE = os.path.dirname(os.path.abspath(__file__))
KS = (0, 1, 2, 4, 8, 12, 16, 20)


def pick_device(name):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def load(path, dev):
    ckpt = torch.load(path, map_location="cpu")
    model = TinyGPT(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(dev).eval(), ckpt


def max_experiments(model):
    # Room left in the context for k experiments plus one probe.
    return (model.c.ctx - 1 - GADGETS) // (GADGETS + 1) - 1


@torch.no_grad()
def tail_logits(model, x, dev, m, chunk=256):
    """Last m positions' logits, in chunks so attention memory stays small."""
    out = []
    for s in range(0, x.shape[0], chunk):
        logits, _ = model(x[s:s + chunk].to(dev))
        out.append(logits[:, -m:, :].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def predict(model, seq, held, value, dev):
    """The model's guess of every reading after a flip, one gadget at a time."""
    x = torch.cat([seq, action_tokens(held, value)], dim=1)
    bits = []
    for i in range(GADGETS):
        last = tail_logits(model, x, dev, 1)[:, 0, :]
        bit = (last[:, 2 * i + 1] > last[:, 2 * i]).long()
        bits.append(bit)
        x = torch.cat([x, (2 * i + bit)[:, None]], dim=1)
    return torch.stack(bits, dim=1)


@torch.no_grad()
def curious_flip(model, seq, rest, dev):
    """Flip the gadget whose outcome the model is least sure about."""
    b, n = rest.shape
    rows = torch.arange(b).repeat_interleave(n)
    held = torch.arange(n).repeat(b)
    every = torch.arange(b * n)
    guess = rest[rows].clone()
    guess[every, held] = 1 - guess[every, held]
    # Score every possible flip in one pass, reading "nothing else changed" as each slot's context.
    x = torch.cat([seq[rows], action_tokens(held, guess[every, held]), reading_tokens(guess)], dim=1)
    slots = tail_logits(model, x, dev, n + 1)[:, :n, :]
    i = torch.arange(n)
    p = F.softmax(torch.stack([slots[:, i, 2 * i], slots[:, i, 2 * i + 1]], dim=-1), dim=-1)
    unsure = -(p * p.clamp_min(1e-9).log()).sum(dim=(1, 2)).view(b, n)
    held = unsure.argmax(dim=1)
    return held, 1 - rest[torch.arange(b), held]


@torch.no_grad()
def probe(model, w, switches, rest, seq, dev):
    """Ask about every possible flip right now and score the guesses."""
    b, n = rest.shape
    rows = torch.arange(b).repeat_interleave(n)
    held = torch.arange(n).repeat(b)
    value = 1 - rest[rows, held]
    truth = w.take(rows).settle(switches[rows], held, value)
    same = rest[rows].clone()
    same[torch.arange(b * n), held] = value  # the lazy guess: "only the flipped gadget changes"
    guess = predict(model, seq[rows], held, value, dev)
    moved = truth != same  # gadgets the flip really changed
    right = guess == truth
    return {
        "exact": right.all(dim=1).float().mean().item(),
        "effects": right[moved].float().mean().item() if moved.any() else float("nan"),
        "lazy": (same == truth).all(dim=1).float().mean().item(),
    }


@torch.no_grad()
def understanding(model, dev, policy="random", worlds=64, ks=KS, seed=999):
    model.eval()
    ks = [k for k in ks if k <= max_experiments(model)]
    gen = torch.Generator().manual_seed(seed)
    w = Worlds(worlds, gen)
    switches, rest, seq = start(w, gen)
    scores = {}
    for k in range(max(ks) + 1):
        if k in ks:
            scores[k] = probe(model, w, switches, rest, seq, dev)
        if k == max(ks):
            break
        if policy == "curious":
            held, value = curious_flip(model, seq, rest, dev)
        else:
            held, value = random_flip(w, rest, gen)
        switches, reading, rest = w.act(switches, held, value)
        seq = torch.cat([seq, action_tokens(held, value), reading_tokens(reading)], dim=1)
    return scores


def table(model, dev, worlds=256):
    r = understanding(model, dev, "random", worlds)
    c = understanding(model, dev, "curious", worlds)
    lines = [
        "| experiments | random flips: all 8 right | random: effects caught | curious flips: all 8 right | curious: effects caught | lazy guess: all 8 right |",
        "|---|---|---|---|---|---|",
    ]
    for k in r:
        lines.append(f"| {k} | {r[k]['exact']:.0%} | {r[k]['effects']:.0%} | {c[k]['exact']:.0%} | "
                     f"{c[k]['effects']:.0%} | {r[k]['lazy']:.0%} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "out", "ckpt.pt"))
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = pick_device(args.device)
    model, ckpt = load(args.ckpt, dev)
    print(f"checkpoint from step {ckpt['step']}")
    print(table(model, dev, args.worlds))


if __name__ == "__main__":
    main()
