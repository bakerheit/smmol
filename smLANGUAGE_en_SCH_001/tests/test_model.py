import json
from pathlib import Path
import tempfile
import unittest

import torch

from model import Config, LanguageModel
from safetensors_io import load_file, save_file
from training_data import build_splits, latest_stage


HERE = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_checked_config_loads_from_the_public_json(self):
        config = Config.from_file(HERE / "model_config.json")
        self.assertEqual((config.vocab_size, config.context_length, config.width), (256, 128, 192))
        self.assertEqual(config.width % config.heads, 0)
        with self.assertRaisesRegex(ValueError, "divisible"):
            Config(width=190, heads=6).validate()

    def test_initialization_is_reproducible_and_forward_shape_is_right(self):
        config = Config(context_length=16, width=48, layers=2, heads=3, dropout=0)
        torch.manual_seed(99)
        first = LanguageModel(config)
        torch.manual_seed(99)
        second = LanguageModel(config)
        self.assertTrue(torch.equal(first.token_embedding.weight, second.token_embedding.weight))
        tokens = torch.randint(0, 256, (2, 12))
        logits, loss = first(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (2, 12, 256))
        self.assertTrue(torch.isfinite(loss))

    def test_safetensors_round_trip_preserves_every_weight(self):
        config = Config(context_length=16, width=48, layers=1, heads=3, dropout=0)
        model = LanguageModel(config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.safetensors"
            save_file(model.state_dict(), path, {"stage": "preschool"})
            loaded, metadata = load_file(path)
        self.assertEqual(metadata["stage"], "preschool")
        self.assertEqual(set(loaded), set(model.state_dict()))
        for name, expected in model.state_dict().items():
            self.assertTrue(torch.equal(loaded[name], expected), name)

    def test_generation_knobs_are_validated(self):
        model = LanguageModel(Config(context_length=16, width=48, layers=1, heads=3, dropout=0))
        tokens = torch.tensor([[65, 66]])
        out = model.generate(tokens, 3, temperature=0, top_k=0, repetition_penalty=1.0)
        self.assertEqual(tuple(out.shape), (1, 5))
        with self.assertRaisesRegex(ValueError, "top_k"):
            model.generate(tokens, 1, top_k=999)


class TrainingDataTests(unittest.TestCase):
    ITEMS = [
        {"level": "preschool", "word": "cat", "sentence": "The cat naps here."},
        {"level": "preschool", "word": "dog", "sentence": "The dog runs home."},
        {"level": "kindergarten", "word": "bright", "sentence": "The bright lamp helps me read."},
    ]

    def test_stage_filter_does_not_train_on_later_rows(self):
        # Pick a seed that puts at least one of these tiny fixtures in each split.
        for seed in range(1000):
            try:
                train, validation, meta = build_splits(self.ITEMS, "preschool", seed)
                break
            except ValueError:
                continue
        else:
            self.fail("could not build a split")
        joined = train + validation
        self.assertNotIn(b"bright", joined)
        self.assertEqual(meta["items"], 2)
        self.assertEqual(latest_stage(self.ITEMS), "kindergarten")


if __name__ == "__main__":
    unittest.main()
