import unittest

import torch

from model import Config, LanguageModel
from prepare import SEPARATOR, complete_stories, split_heldout
from train import sample_batch


class CorpusTests(unittest.TestCase):
    def test_complete_stories_drops_partial_tail_and_markers(self):
        raw = b"first" + SEPARATOR + b"second" + SEPARATOR + b"partial"
        self.assertEqual(complete_stories(raw), b"first\n\nsecond\n")

    def test_heldout_split_is_story_disjoint(self):
        raw = SEPARATOR.join([b"alpha", b"bravo", b"charlie", b"delta", b""])
        val, test = split_heldout(raw)
        self.assertEqual(val, b"alpha\n\nbravo\n")
        self.assertEqual(test, b"charlie\n\ndelta\n")
        val_lines = {line for line in val.splitlines() if line}
        test_lines = {line for line in test.splitlines() if line}
        self.assertFalse(val_lines & test_lines)


class ModelTests(unittest.TestCase):
    def test_forward_loss_and_generation_shapes(self):
        config = Config(ctx=16, d=24, layers=2, heads=4, dropout=0.0)
        model = LanguageModel(config)
        x = torch.randint(0, 256, (3, 16))
        logits, loss = model(x, x)
        self.assertEqual(logits.shape, (3, 16, 256))
        self.assertTrue(torch.isfinite(loss))
        generated = model.generate(x[:, :4], 5, temperature=1.0, top_k=10)
        self.assertEqual(generated.shape, (3, 9))

    def test_batch_targets_are_next_bytes(self):
        source = torch.arange(100, dtype=torch.uint8)
        generator = torch.Generator().manual_seed(1)
        x, y = sample_batch(source, 4, 12, generator, "cpu")
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))


if __name__ == "__main__":
    unittest.main()
