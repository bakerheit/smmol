"""Score each trained variant on +, - and * at every length from 1 to 10 digits.

Training numbers had 1 to 6 digits, so 7 to 10 are lengths no variant has ever seen. Every variant gets
exactly the same problems.

    python3 evaluate.py
    python3 evaluate.py --per 500
"""
import argparse
import json
import os
import random
import sys
import time

import torch

from model import VARIANTS, Config, MathGPT
from problems import OPS, decode, encode, expected, problem, question, readable, solve

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIGITS = range(1, 11)
NAMES = {"+": "Addition", "-": "Subtraction", "*": "Multiplication"}


def load(path, device):
    ck = torch.load(path, map_location="cpu")
    model = MathGPT(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def accuracy(model, op, digits, n, rng, device, keep_mistakes=0):
    """Share of n problems (both numbers exactly `digits` long) answered exactly right, and a few misses."""
    reverse = model.c.reverse
    probs = [problem(digits, op, digits, rng) for _ in range(n)]
    ids = torch.tensor([encode(question(*p, reverse)) for p in probs], device=device)
    want = [expected(*p, reverse) for p in probs]
    was_training = model.training
    model.eval()
    out = model.answer(ids, max(map(len, want)) + 1)
    model.train(was_training)
    right, mistakes = 0, []
    for p, w, row in zip(probs, want, out[:, ids.shape[1]:].tolist()):
        got = decode(row)
        got = got[:got.index("$") + 1] if "$" in got else got
        if got == w:
            right += 1
        elif len(mistakes) < keep_mistakes:
            mistakes.append({"problem": "%d%s%d" % p, "right": solve(*p), "got": readable(got, reverse) or "(nothing)"})
    return right / n, mistakes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=300, help="problems per operation and length")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    found = [v for v in VARIANTS if os.path.exists(os.path.join(args.out, v + ".pt"))]
    if not found:
        sys.exit("no checkpoints in %s yet: run train.py first" % args.out)
    results = {"when": time.strftime("%Y-%m-%d %H:%M"), "per": args.per, "variants": {}}
    for v in found:
        model, ck = load(os.path.join(args.out, v + ".pt"), args.device)
        entry = {k: ck.get(k) for k in ("params", "step", "problems", "seconds", "loss", "max_digits", "offset_max")}
        entry["accuracy"], entry["mistakes"] = {}, {}
        started = time.time()
        for op in OPS:
            row = []
            for digits in TEST_DIGITS:
                rng = random.Random(1000 * digits + OPS.index(op))  # the same problems for every variant
                acc, missed = accuracy(model, op, digits, args.per, rng, args.device, keep_mistakes=3)
                row.append(round(acc, 3))
                if missed and digits in (ck["max_digits"], ck["max_digits"] + 2):
                    entry["mistakes"]["%s%d" % (op, digits)] = missed
            entry["accuracy"][op] = row
        print("scored %s in %.0fs" % (v, time.time() - started), flush=True)
        results["variants"][v] = entry

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)

    trained = results["variants"][found[0]]["max_digits"]
    for op in OPS:
        print("\n%s: exactly right, out of %d problems per length (trained on 1 to %d digits)" % (NAMES[op], args.per, trained))
        print("| variant | " + " | ".join(("%d" if d <= trained else "**%d**") % d for d in TEST_DIGITS) + " |")
        print("|---" * (len(TEST_DIGITS) + 1) + "|")
        for v, e in results["variants"].items():
            print("| %s | %s |" % (v, " | ".join("%d%%" % round(100 * a) for a in e["accuracy"][op])))
    print("\nSome misses (right answer, then what it wrote):")
    for v, e in results["variants"].items():
        for key, missed in e["mistakes"].items():
            m = missed[0]
            print("  %-8s %-4s %s = %s, wrote %s" % (v, key, m["problem"], m["right"], m["got"]))


if __name__ == "__main__":
    main()
