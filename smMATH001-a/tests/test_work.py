"""smMATH001-a: worked solutions are right, readable, and within their sizes; the model's cached writing matches a full
forward pass; and the harness solver reads answers back correctly.

    python3 -m unittest discover -s tests
"""
import os
import random
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict
from fractions import Fraction

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import work as W  # noqa: E402
from model import END, SEP, VOCAB, Config, MathGPT, batch, encode, places  # noqa: E402


def exact(tree):
    if isinstance(tree, str):
        return Fraction(tree)
    op, a, b = tree
    x, y = exact(a), exact(b)
    return x + y if op == "+" else x - y if op == "-" else x * y if op == "*" else x / y


def has(tree, chars):
    return any(ch in W.render(tree) for ch in chars)


class WorkTests(unittest.TestCase):
    def test_small_worked_solutions(self):
        self.assertEqual(W.solve_tree(W.parse("2+2")), ("002+002=004", 400))
        self.assertEqual(W.question(W.parse("(12+8)/5")), "(0021+008)/005")
        work, answer = W.solve_tree(W.parse("(12+8)/5"))
        self.assertEqual((W.final_answer(work), W.unscaled(answer)), (400, "4"))
        self.assertEqual(W.readable(work), ["12 + 8 = 20", "20 ÷ 5:", "  5 × 4 = 20", "  20 − 20 = 0", "  = 4"])
        self.assertEqual(W.unscaled(W.solve_tree(W.parse("12.5+7.25"))[1]), "19.75")
        self.assertEqual(W.unscaled(W.solve_tree(W.parse("3-5"))[1]), "-2")
        with self.assertRaises(ValueError):
            W.solve_tree(W.parse("(3-5)*2"))  # below zero partway through
        for bad in ("2**3", "sqrt(4)", "1.609344*2", "-3+4", "x+1"):
            with self.assertRaises(ValueError, msg=bad):
                W.parse(bad)

    def test_answers_match_exact_arithmetic(self):
        self.assertEqual(W.solve_tree(W.parse("322234*21323*212231"))[1], 322234 * 21323 * 212231 * 100)
        for e in W.generate(3000, seed=4):
            tree, value = W.parse(e["expression"]), Fraction(e["answer"], 100)
            if not has(tree, "/."):
                self.assertEqual(value, exact(tree), e["expression"])
            elif tree[0] == "/" and not has(tree[1], "/.") and not has(tree[2], "/."):
                self.assertTrue(0 <= exact(tree) - value < Fraction(1, 100), e["expression"])  # cut, never rounded up

    def test_generated_examples_stay_within_their_sizes(self):
        examples = W.generate(2000, seed=5)
        self.assertEqual(examples[:3], W.generate(3, seed=5))
        self.assertTrue(all(len(e["work"]) <= W.MAX_WORK and len(e["question"]) <= W.MAX_QUESTION for e in examples))
        self.assertEqual({e["kind"] for e in examples}, {"add", "subtract", "multiply", "divide", "mixed"})
        self.assertTrue(any(e["decimals"] for e in examples) and any(e["harness"] for e in examples))
        for e in examples[:300]:
            self.assertEqual(W.final_answer(e["work"]), e["answer"])
            self.assertFalse([line for line in W.readable(e["work"]) if "=" not in line and not line.endswith(":")])


class ModelTests(unittest.TestCase):
    def test_only_the_work_is_scored_and_places_count_inside_numbers(self):
        x, y, where = batch([{"question": "002+002", "work": "002+002=004"}])
        scored = [t for t in y[0].tolist() if t != -100]
        self.assertEqual("".join(VOCAB[t] for t in scored[:-1]) + "|" + str(scored[-1] == END), "002+002=004|True")
        self.assertEqual(x[0].tolist().index(SEP), 7)
        self.assertEqual(places(torch.tensor([encode("0021+008")]))[0].tolist()[:9], [1, 2, 3, 4, 0, 1, 2, 3, 0])

    def test_cached_writing_matches_a_full_forward_pass(self):
        torch.manual_seed(0)
        model = MathGPT(Config(d=32, layers=2, heads=4, max_len=128)).eval()
        question = "0021+008"
        written = model.work(question, max_new=20)
        ids = encode(question)
        for ch in written:
            logits, _ = model(torch.tensor([ids]))
            self.assertEqual(VOCAB[int(logits[0, -1].argmax())], ch)
            ids.append(VOCAB.index(ch))

    def test_the_harness_solver_reads_the_answer_and_work(self):
        from solve import Solver
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        model = MathGPT(Config(d=32, layers=1, heads=4, max_len=256))
        path = os.path.join(folder, "math.pt")
        torch.save({"model": model.state_dict(), "config": asdict(model.c), "step": 0}, path)
        solver = Solver(path)
        solver.model.work = lambda q: W.solve_tree(W.parse("(12+8)/5"))[0]
        result = solver.solve("(12+8)/5")
        self.assertEqual((result["answer"], result["work"][0]), ("4", "12 + 8 = 20"))
        self.assertTrue(solver.fits("2340*17.5/100"))
        self.assertFalse(solver.fits("26.2*1.609344"))


if __name__ == "__main__":
    unittest.main()
