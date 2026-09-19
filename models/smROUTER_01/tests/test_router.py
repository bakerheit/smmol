"""smROUTER_01: the data keeps its labels honest, encoding and switching behave, and the rules baseline works.

    python3 -m unittest discover -s tests
"""
import json
import os
import re
import sys
import unittest
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data  # noqa: E402
import rules  # noqa: E402
import train  # noqa: E402
from model import CLS, PAD, SEP, RouterNet, encode, encode_batch, only_allowed  # noqa: E402


class DataTests(unittest.TestCase):
    def test_generated_labels_are_valid_and_every_label_shows_up(self):
        examples = data.generate(5000, seed=1)
        self.assertEqual(examples[:5], data.generate(5, seed=1))  # same seed, same data
        intents = Counter(e["intent"] for e in examples)
        tools = Counter(e["tool"] for e in examples)
        self.assertEqual(set(intents), set(data.LABELS["intents"]))
        self.assertEqual(set(tools), set(data.LABELS["tools"]))
        self.assertEqual({e["ask_first"] for e in examples}, {True, False})
        self.assertFalse([e for e in examples if "{" in e["text"] or "{" in e["prev"]])  # every slot got filled
        bare = [e["text"] for e in examples if e["tool"] == "calculator" and not re.search(r"[a-z]{3}", e["text"].lower())]
        self.assertGreater(len(bare), 100)  # plenty of arithmetic with no words around it

    def test_answers_always_follow_a_question(self):
        for e in data.generate(3000, seed=2):
            if e["intent"] == "answer":
                self.assertTrue(e["prev"].endswith("?"), e)

    def test_hand_written_test_set_uses_known_labels_and_never_reaches_training(self):
        with open(os.path.join(ROOT, "test.json")) as f:
            test = json.load(f)
        with open(os.path.join(ROOT, "test_arithmetic.json")) as f:
            test += json.load(f)
        for e in test:
            self.assertIn(e["intent"], data.LABELS["intents"])
            self.assertIn(e["tool"], data.LABELS["tools"])
            self.assertIsInstance(e["ask_first"], bool)
        training = {data.normal(e["text"]) for e in data.without(data.generate(60000, seed=0), test)}
        self.assertEqual([e["text"] for e in test if data.normal(e["text"]) in training], [])


class ModelTests(unittest.TestCase):
    def test_encoding_keeps_the_end_of_the_question(self):
        prev = "x" * 100 + " Which city?"
        row = encode(prev, "Denver", 192)
        self.assertEqual(row[0], CLS)
        self.assertEqual(bytes(row[1:row.index(SEP)]).decode().endswith("Which city?"), True)
        ids, pad = encode_batch([("", "hi"), (prev, "Denver")], 192)
        self.assertEqual((ids[0, 4].item(), pad[0, 4].item()), (PAD, True))

    def test_switched_off_tools_get_no_probability(self):
        logits = torch.tensor([[0.0, 5.0, 1.0]])
        allowed = torch.tensor([[True, False, True]])
        probs = torch.softmax(only_allowed(logits, allowed), 1)
        self.assertLess(probs[0, 1].item(), 1e-6)
        self.assertEqual(probs[0].argmax().item(), 2)

    def test_random_switches_point_missing_tools_to_none(self):
        tool = torch.tensor([1, 2, 3, 0] * 50)
        allowed, target = train.random_switches(tool, torch.Generator().manual_seed(0), keep=0.5)
        rows = torch.arange(len(tool))
        self.assertTrue(allowed[:, 0].all())
        self.assertTrue((target[~allowed[rows, tool]] == 0).all())
        self.assertTrue((target[allowed[rows, tool]] == tool[allowed[rows, tool]]).all())

    def test_network_shapes(self):
        net = RouterNet(6, 7, d=48, layers=1, heads=4, max_len=64)
        ids, pad = encode_batch([("", "what's 2+2"), ("Which city?", "Oslo")], 64)
        li, lt, la = net(ids, pad)
        self.assertEqual((li.shape, lt.shape, la.shape), ((2, 6), (2, 7), (2, 2)))

    def test_scoring_counts_off_tools_as_none(self):
        examples = [{"prev": "", "text": "weather", "intent": "question", "tool": "web_search", "ask_first": False}]
        guess = [{"intent": "question", "tool": "none", "ask_first": False}]
        self.assertEqual(train.score(examples, guess, off=("web_search",))[0]["all"], 1.0)
        self.assertEqual(train.score(examples, guess)[0]["all"], 0.0)


class RulesTests(unittest.TestCase):
    def test_obvious_cases(self):
        self.assertEqual(rules.classify("", "lol ok")["intent"], "small_talk")
        self.assertEqual(rules.classify("", "what's 20% of 80")["tool"], "calculator")
        self.assertEqual(rules.classify("", "summarize https://example.com/a")["tool"], "web_browser")
        self.assertEqual(rules.classify("", "remember that my vet is Dr. Cho")["intent"], "remember")
        self.assertEqual(rules.classify("Which city?", "Denver")["intent"], "answer")


if __name__ == "__main__":
    unittest.main()
