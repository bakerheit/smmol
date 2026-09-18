"""smMATH001-a as a harness module: a whole expression in, the worked steps and the answer out.

    python3 solve.py "(12+8)/5"
    python3 solve.py "322234*21323*212231"
"""
import argparse
import os
import time
from fractions import Fraction

import torch

from model import Config, MathGPT
from work import final_answer, parse, question, readable, render, scaled, unscaled

HERE = os.path.dirname(os.path.abspath(__file__))


class Solver:
    takes = "expression"  # the harness hands it whole expressions, not single steps

    def __init__(self, path):
        ck = torch.load(path, map_location="cpu")
        self.model = MathGPT(Config(**ck["config"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.steps = ck.get("step")

    def fits(self, expression):
        """Whether it can take the expression at all: numbers with up to 2 decimal places, + - * / and brackets."""
        try:
            return len(question(parse(expression))) <= 200
        except ValueError:
            return False

    def solve(self, expression):
        """{"answer": "4" or None, "work": readable lines, "raw": what it wrote}."""
        tree = parse(expression)
        if isinstance(tree, str):
            return {"answer": unscaled(scaled(tree)), "work": [], "raw": ""}
        raw = self.model.work(question(tree))
        answer = final_answer(raw)
        return {"answer": None if answer is None else unscaled(answer), "work": readable(raw), "raw": raw}


def exact(tree):
    if isinstance(tree, str):
        return Fraction(tree)
    op, a, b = tree
    x, y = exact(a), exact(b)
    return x + y if op == "+" else x - y if op == "-" else x * y if op == "*" else x / y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("expression")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    solver = Solver(os.path.join(args.out, "math.pt"))
    started = time.perf_counter()
    result = solver.solve(args.expression)
    seconds = time.perf_counter() - started
    tree = parse(args.expression)
    print("%s = %s   (exact: %s, %.2fs, %d characters of work)" % (render(tree), result["answer"], float(exact(tree)), seconds, len(result["raw"])))
    for line in result["work"]:
        print("  " + line)


if __name__ == "__main__":
    main()
