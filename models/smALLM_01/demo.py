"""Watch smALLM_01 poke at one brand-new world, then reveal the secret wiring."""
import argparse
import os

import torch

from evaluate import curious_flip, load, pick_device, predict
from world import LETTERS, Worlds, action_tokens, random_flip, reading_tokens, start

HERE = os.path.dirname(os.path.abspath(__file__))


def bits(row):
    return "".join(str(int(v)) for v in row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "out", "ckpt.pt"))
    ap.add_argument("--experiments", type=int, default=16)
    ap.add_argument("--policy", choices=("random", "curious"), default="curious")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = pick_device(args.device)
    model, _ = load(args.ckpt, dev)
    gen = torch.Generator().manual_seed(args.seed)
    w = Worlds(1, gen)
    switches, rest, seq = start(w, gen)

    print(f"{'':16}{LETTERS}")
    print(f"{'start':16}{bits(rest[0])}")
    for t in range(args.experiments):
        if args.policy == "curious":
            held, value = curious_flip(model, seq, rest, dev)
        else:
            held, value = random_flip(w, rest, gen)
        guess = predict(model, seq, held, value, dev)
        switches, reading, rest = w.act(switches, held, value)
        wrong = int((guess != reading).sum())
        verdict = "right" if wrong == 0 else f"{wrong} wrong"
        print(f"{t + 1:2d}. flip {LETTERS[held[0]]} to {int(value[0])}  guess {bits(guess[0])}  saw {bits(reading[0])}  {verdict}")
        seq = torch.cat([seq, action_tokens(held, value), reading_tokens(reading)], dim=1)

    print("\nSecret wiring:")
    for line in w.describe(0):
        print(f"  {line}")


if __name__ == "__main__":
    main()
