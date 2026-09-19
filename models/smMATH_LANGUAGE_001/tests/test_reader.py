"""smMATH_LANGUAGE_001: generated targets always work out, hand-written messages stay out of training, and only the
written problems are scored.

    python3 -m unittest discover -s tests
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data  # noqa: E402
from model import SEP, Config, ReaderGPT, batch  # noqa: E402
from score import evaluate, grade, values  # noqa: E402


class DataTests(unittest.TestCase):
    def test_every_generated_target_works_out(self):
        examples = data.generate(4000, seed=1)
        self.assertEqual(examples[:5], data.generate(5, seed=1))
        for e in examples:
            values([p["expression"] for p in data.parse_target(e["target"])])  # raises if a target is broken
        self.assertTrue(any(not e["target"] for e in examples))          # messages with nothing to work out
        self.assertTrue(any("p1" in e["target"] for e in examples))      # follow-ups that use an earlier answer
        self.assertTrue(any(";" in e["target"] for e in examples))       # several problems in one message

    def test_targets_become_the_harness_problem_list(self):
        self.assertEqual(data.parse_target("3*4*5|cubic feet|volume;p1/27|cubic yards|volume"),
                         [{"id": "p1", "expression": "3*4*5", "unit": "cubic feet", "about": "volume"},
                          {"id": "p2", "expression": "p1/27", "unit": "cubic yards", "about": "volume"}])
        self.assertEqual(data.parse_target(""), [])

    def test_hand_written_messages_work_out_and_never_reach_training(self):
        with open(os.path.join(ROOT, "test.json")) as f:
            test = json.load(f)
        for e in test:
            values(e["expected"])
        training = {data.normal(e["text"]) for e in data.without(data.generate(30000, seed=0), test)}
        self.assertEqual([e["text"] for e in test if data.normal(e["text"]) in training], [])


class ScoreTests(unittest.TestCase):
    def test_grading_compares_values_not_spelling(self):
        self.assertEqual(grade(["15/100*80"], ["80*15/100"]), (True, True))
        self.assertEqual(grade(["3*4*5", "p1/27"], ["3*4*5", "p1/27"]), (True, True))
        self.assertEqual(grade(["60/27"], ["3*4*5", "p1/27"]), (False, True))  # right answer, one problem short
        self.assertEqual(grade([], []), (True, True))
        self.assertEqual(grade(["2+2"], []), (False, False))                  # found math that isn't there
        self.assertEqual(grade(["p2+1"], ["3"]), (False, False))              # a broken reference is simply wrong
        with self.assertRaises(ValueError):
            evaluate("__import__('os')")


class ModelTests(unittest.TestCase):
    def test_only_the_written_problems_are_scored(self):
        x, y = batch([{"text": "2+2", "target": "2+2||"}])
        scored = [t for t in y[0].tolist() if t != -100]
        self.assertEqual(bytes(scored[:-1]).decode(), "2+2||")  # and then <END>
        self.assertEqual(x[0].tolist().index(SEP), 3)

    def test_the_model_runs_and_the_harness_reader_returns_problems(self):
        model = ReaderGPT(Config(d=32, layers=1, heads=4))
        x, y = batch([{"text": "what's 2+2", "target": "2+2||"}, {"text": "lol", "target": ""}])
        self.assertTrue(torch.isfinite(model(x, y)[1]))
        self.assertIsInstance(model.read("2+2"), str)

        from read import Reader
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        path = os.path.join(folder, "reader.pt")
        torch.save({"model": model.state_dict(), "config": asdict(model.c), "step": 0}, path)
        reader = Reader(path)
        reader.model.read = lambda text: "3*4*5|cubic feet|volume;p1/27|cubic yards|volume"
        self.assertEqual([p["id"] for p in reader.read("volume of 3ft x 4ft x 5ft, and in cubic yards?")], ["p1", "p2"])


if __name__ == "__main__":
    unittest.main()
