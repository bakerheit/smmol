"""Ask every trained smMATH01 variant one problem.

    python3 ask.py 48213+9977
    python3 ask.py "1234567 * 89"
"""
import argparse
import os
import re
import sys

import torch

from evaluate import load
from model import VARIANTS
from problems import decode, encode, expected, question, readable, solve

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()

    m = re.fullmatch(r"(\d+)([+\-*x×])(\d+)", "".join(args.problem).replace(",", "").replace(" ", ""))
    if not m:
        sys.exit("write it like 48213+9977, 900-35 or 1234*56")
    a, op, b = int(m.group(1)), {"x": "*", "×": "*"}.get(m.group(2), m.group(2)), int(m.group(3))
    if op == "-" and b > a:
        sys.exit("it only learned subtraction with an answer of 0 or more, so put the bigger number first")
    found = [v for v in VARIANTS if os.path.exists(os.path.join(args.out, v + ".pt"))]
    if not found:
        sys.exit("no checkpoints in %s yet: run train.py first" % args.out)

    print("%d %s %d = %d" % (a, op, b, solve(a, op, b)))
    longest = max(len(str(a)), len(str(b)))
    for v in found:
        model, ck = load(os.path.join(args.out, v + ".pt"), "cpu")
        ids = torch.tensor([encode(question(a, op, b, model.c.reverse))])
        want = expected(a, op, b, model.c.reverse)
        if ids.shape[1] + len(want) > model.c.ctx:
            sys.exit("that's too long for its %d-token window" % model.c.ctx)
        out = decode(model.answer(ids, len(want) + 1)[0, ids.shape[1]:].tolist())
        got = out[:out.index("$") + 1] if "$" in out else out
        print("  %-9s %-24s %s" % (v, readable(got, model.c.reverse) or "(nothing)", "right" if got == want else "wrong"))
    if longest > ck["max_digits"]:
        print("(%d digits is longer than any number it trained on)" % longest)


if __name__ == "__main__":
    main()
