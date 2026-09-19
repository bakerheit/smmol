"""Offline checks for the data, model plumbing, grounding rules, and held-out cases."""
import json
import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import data  # noqa: E402
from model import END, SEP, Config, LanguageGPT, batch, encode  # noqa: E402
from score import grade  # noqa: E402


class DataTests(unittest.TestCase):
    def test_generated_lessons_are_repeatable_bounded_and_grounded(self):
        rows = data.generate(5000, seed=3)
        self.assertEqual(rows[:10], data.generate(10, seed=3))
        self.assertEqual(set(e["kind"] for e in rows), {"math", "memory", "ask", "file", "file_error",
                                                         "web", "need", "direct", "assumption", "unavailable", "copy"})
        for example in rows:
            self.assertLessEqual(len(encode(example["prompt"], example["reply"])), 1024)
            self.assertEqual(grade(example, example["reply"]), (True, []))

    def test_handwritten_dev_and_test_are_separate_and_grounded(self):
        sets = []
        for name in ("dev.json", "test.json"):
            with open(os.path.join(ROOT, name)) as f:
                rows = json.load(f)
            cooked = [dict(row, prompt=data.compact(row["payload"])) for row in rows]
            for example in cooked:
                self.assertEqual(grade(example, example["reply"]), (True, []))
            sets.append({e["prompt"] for e in cooked})
        self.assertFalse(sets[0] & sets[1])

    def test_compact_drops_noise_and_caps_long_results(self):
        prompt = data.compact({"message": "  hello\nthere  ", "unused": "no", "memories": ["x" * 500] * 8,
                               "tool_results": [{"tool": "web_browser", "ok": True, "result": "z" * 900}]})
        value = json.loads(prompt)
        self.assertEqual(value["message"], "hello there")
        self.assertNotIn("unused", value)
        self.assertEqual((len(value["memories"]), len(value["memories"][0]), len(value["tool_results"][0]["result"])),
                         (3, 220, 520))


class ModelTests(unittest.TestCase):
    def test_only_reply_bytes_are_scored(self):
        x, y = batch([{"prompt": "abc", "reply": "yes"}])
        self.assertEqual(x[0].tolist().index(SEP), 3)
        scored = [v for v in y[0].tolist() if v != -100]
        self.assertEqual(scored, list(b"yes") + [END])

    def test_a_tiny_model_runs(self):
        model = LanguageGPT(Config(max_len=128, d=32, layers=1, heads=4))
        x, y = batch([{"prompt": "message", "reply": "reply"}])
        logits, loss = model(x, y)
        self.assertEqual(logits.shape[:2], x.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.param_count(), 0)


class ScoreTests(unittest.TestCase):
    def example(self, kind="web"):
        return {"kind": kind, "prompt": data.compact({"message": "Weather in Boston",
                "tool_results": [{"tool": "web_browser", "ok": True,
                                  "target": "https://example.test/dayton", "result": "It is 72°F."}]}),
                "reply": "It is 72°F."}

    def test_invented_numbers_and_urls_fail(self):
        example = self.example()
        self.assertFalse(grade(example, "It is 81°F.")[0])
        self.assertFalse(grade(example, "See https://wrong.test/now")[0])
        self.assertTrue(grade(example, "It is 72°F.")[0])

    def test_ask_and_need_shapes_are_strict(self):
        ask = self.example("ask")
        ask["reply"] = "I need the city. Which city?"
        self.assertTrue(grade(ask, "I need the city. Which city?")[0])
        self.assertFalse(grade(ask, "Which city? What date?")[0])
        need = self.example("need")
        need["reply"] = "NEED: web_browser https://example.test/dayton"
        self.assertTrue(grade(need, "NEED: web_browser https://example.test/dayton")[0])
        self.assertFalse(grade(need, "I'll check https://example.test/dayton")[0])


if __name__ == "__main__":
    unittest.main()
