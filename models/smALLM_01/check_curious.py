"""Is the 'curious' picker choosing flips it doesn't understand, or just flips with big effects?"""
import os

import torch

from evaluate import curious_flip, load, pick_device
from world import GADGETS, Worlds, action_tokens, random_flip, reading_tokens, start

HERE = os.path.dirname(os.path.abspath(__file__))
WORLDS, STEPS = 256, 20


def effects(w, switches, rest):
    """How many other gadgets each possible flip would change right now."""
    b, n = rest.shape
    rows = torch.arange(b).repeat_interleave(n)
    held = torch.arange(n).repeat(b)
    value = 1 - rest[rows, held]
    truth = w.take(rows).settle(switches[rows], held, value)
    same = rest[rows].clone()
    same[torch.arange(b * n), held] = value
    return (truth != same).sum(1).view(b, n)


def main():
    dev = pick_device("auto")
    model, _ = load(os.path.join(HERE, "out", "ckpt.pt"), dev)
    for policy in ("random", "curious"):
        gen = torch.Generator().manual_seed(999)
        w = Worlds(WORLDS, gen)
        switches, rest, seq = start(w, gen)
        rows = torch.arange(WORLDS)
        tried = torch.zeros(WORLDS, GADGETS, dtype=torch.bool)
        picked, typical, biggest, repeat = [], [], [], []
        prev = None
        for _ in range(STEPS):
            eff = effects(w, switches, rest)
            if policy == "curious":
                held, value = curious_flip(model, seq, rest, dev)
            else:
                held, value = random_flip(w, rest, gen)
            picked.append(eff[rows, held].float().mean().item())
            typical.append(eff.float().mean().item())
            biggest.append((eff[rows, held] == eff.max(1).values).float().mean().item())
            if prev is not None:
                repeat.append((held == prev).float().mean().item())
            prev = held
            tried[rows, held] = True
            switches, reading, rest = w.act(switches, held, value)
            seq = torch.cat([seq, action_tokens(held, value), reading_tokens(reading)], dim=1)
        avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
        print(f"{policy:8s} | other gadgets its flips changed: {avg(picked):.2f} (typical flip {avg(typical):.2f}) | "
              f"picked a biggest-effect flip {avg(biggest):.0%} | same gadget twice in a row {avg(repeat):.0%} | "
              f"gadgets tried in {STEPS} flips: {tried.sum(1).float().mean():.1f} of {GADGETS}")


if __name__ == "__main__":
    main()
