"""smMATH01 as a harness module: one whole-number problem in, the digits it writes out.

    from solve import Solver
    Solver("out/abacus.pt").answer(48213, "+", 9977)   # "58190", or None if it never finished a number
"""
import torch

from evaluate import load
from problems import OPS, decode, encode, question, readable


class Solver:
    def __init__(self, path):
        self.model, ck = load(path, "cpu")
        self.variant = self.model.c.variant
        self.max_digits = ck["max_digits"]
        self.steps = ck["step"]

    def fits(self, a, op, b):
        """Whether it can take the problem at all: +, - or *, no negatives, and room in its window for any answer."""
        return (op in OPS and a >= 0 and b >= 0 and (op != "-" or a >= b)
                and len(question(a, op, b, False)) + len(str(a)) + len(str(b)) + 1 <= self.model.c.ctx)

    def answer(self, a, op, b):
        """The digits it writes for a op b, in the usual order, or None if it doesn't finish a whole number."""
        if not self.fits(a, op, b):
            raise ValueError("smMATH01 can't take %d %s %d" % (a, op, b))
        ids = torch.tensor([encode(question(a, op, b, self.model.c.reverse))])
        out = decode(self.model.answer(ids, self.model.c.ctx - ids.shape[1])[0, ids.shape[1]:].tolist())
        if "$" not in out:
            return None
        digits = readable(out[:out.index("$") + 1], self.model.c.reverse)
        return digits if digits.isdigit() else None
