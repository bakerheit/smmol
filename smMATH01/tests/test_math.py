"""smMATH01: the problems are right, only answers are scored, place labels count correctly, and a tiny model learns.

    python3 -m unittest discover -s tests
"""
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import problems as P  # noqa: E402
from evaluate import accuracy  # noqa: E402
from model import VARIANTS, Config, MathGPT  # noqa: E402


class ProblemTests(unittest.TestCase):
    def test_lines_in_both_orders(self):
        self.assertEqual(P.line(48213, "+", 9977, False), "48213+9977=58190$")
        self.assertEqual(P.line(48213, "+", 9977, True), "31284+7799=09185$")
        self.assertEqual(P.line(100, "-", 1, True), "001-1=99$")
        self.assertEqual(P.line(12, "*", 34, False), "12*34=408$")
        self.assertEqual(P.decode(P.encode("31284+7799=09185$") + [P.PAD]), "31284+7799=09185$")
        self.assertEqual(P.readable("09185$", True), "58190")

    def test_numbers_have_exact_lengths_and_subtraction_stays_non_negative(self):
        rng = random.Random(0)
        for digits in range(1, 11):
            for _ in range(200):
                self.assertEqual(len(str(P.number(digits, rng))), digits)
                a, _, b = P.problem(digits, "-", digits, rng)
                self.assertGreaterEqual(a, b)

    def test_places_count_inside_each_number(self):
        self.assertEqual(P.places(torch.tensor([P.encode("312+79=")]))[0].tolist(), [1, 2, 3, 0, 1, 2, 0])

    def test_batches_score_only_the_right_answer_and_stay_short(self):
        x, y, _ = P.batch(128, random.Random(1), reverse=True)
        for xi, yi in zip(x.tolist(), y.tolist()):
            a, op, b = re.fullmatch(r"(\d+)([+\-*])(\d+)", P.decode(xi).split("=")[0]).groups()
            self.assertLessEqual(max(len(a), len(b)), P.TRAIN_DIGITS)
            a, b = int(a[::-1]), int(b[::-1])
            self.assertEqual(P.decode([t for t in yi if t != -100]), P.expected(a, op, b, True))

    def test_a_shift_moves_a_whole_line_together(self):
        x, _, where = P.batch(32, random.Random(2), reverse=True, offset_max=25)
        for shifted, base in zip(where.tolist(), P.places(x).tolist()):
            shifts = {s - b for s, b in zip(shifted, base) if b > 0}
            self.assertEqual(len(shifts), 1)
            self.assertTrue(0 <= shifts.pop() <= 25)
            self.assertTrue(all(s == 0 for s, b in zip(shifted, base) if b == 0))


class ModelTests(unittest.TestCase):
    def test_every_variant_runs(self):
        x, y, where = P.batch(8, random.Random(3), reverse=True)
        for v in VARIANTS:
            model = MathGPT(Config(variant=v, d=32, layers=1, heads=4))
            logits, loss = model(x, where if v == "abacus" else None, y)
            self.assertEqual(tuple(logits.shape), (8, x.shape[1], P.VOCAB_SIZE))
            self.assertTrue(torch.isfinite(loss))
            self.assertLessEqual(model.answer(torch.tensor([P.encode("21+43=")]), 5).shape[1], 6 + 5)

    def test_a_tiny_model_learns_one_digit_addition(self):
        torch.manual_seed(0)
        rng = random.Random(0)
        model = MathGPT(Config(variant="abacus", d=64, layers=2, heads=4))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(400):
            x, y, where = P.batch(128, rng, reverse=True, max_digits=1, ops="+", offset_max=3)
            loss = model(x, where, y)[1]
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc, _ = accuracy(model, "+", 1, 200, random.Random(9), "cpu")
        self.assertGreater(acc, 0.9)

    def test_the_harness_solver_reads_answers_back_in_the_usual_order(self):
        from solve import Solver
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        model = MathGPT(Config(variant="abacus", d=32, layers=1, heads=4))
        path = os.path.join(folder, "abacus.pt")
        torch.save({"model": model.state_dict(), "config": asdict(model.c), "step": 0, "max_digits": 6}, path)
        solver = Solver(path)
        writes = lambda text: (lambda ids, n: torch.cat([ids, torch.tensor([P.encode(text)])], dim=1))  # noqa: E731
        solver.model.answer = writes("09185$")
        self.assertEqual(solver.answer(48213, "+", 9977), "58190")
        solver.model.answer = writes("091")
        self.assertIsNone(solver.answer(48213, "+", 9977))  # never wrote "$"
        self.assertFalse(solver.fits(3, "-", 5))
        self.assertFalse(solver.fits(10 ** 30, "*", 10 ** 30))
        self.assertTrue(solver.fits(123456, "*", 654321))


if __name__ == "__main__":
    unittest.main()
