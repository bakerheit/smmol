import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from training_data import build_splits, build_supervised_sets, encode_supervised, prompt, read_examples


class ConversationDataTests(unittest.TestCase):
    def test_prompt_matches_harness_shape(self):
        item = {"user": "What is it?", "known": ["it is blue"], "assistant": "It is blue."}
        self.assertEqual(
            prompt(item),
            "Question: What is it?\nKnown information: it is blue\nHelpful English reply: ",
        )

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            item = {"id": "same", "category": "x", "user": "Hi", "assistant": "Hello"}
            path.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_examples(path)

    def test_real_corpus_has_stable_splits(self):
        examples = read_examples(HERE / "data" / "conversations.jsonl")
        train, validation, metadata = build_splits(examples)
        self.assertGreaterEqual(metadata["items"], 800)
        self.assertGreater(len(train), len(validation))
        self.assertGreater(metadata["validation_items"], 50)

    def test_supervised_encoding_masks_the_prompt(self):
        item = {"id": "one", "category": "social", "user": "Hi", "assistant": "Hello"}
        inputs, targets = encode_supervised(item, 64)
        prefix_bytes = len(prompt(item).encode("utf-8"))
        self.assertTrue(all(value == -100 for value in targets[: prefix_bytes - 1]))
        self.assertEqual(bytes(value for value in targets if value >= 0).decode("utf-8"), "Hello\n\n")
        self.assertEqual(len(inputs), 64)

    def test_conversations_are_weighted_over_school_replay(self):
        examples = read_examples(HERE / "data" / "conversations.jsonl")
        train, validation, metadata = build_supervised_sets(examples, 128)
        self.assertEqual(metadata["conversation_weight"], 4)
        self.assertGreater(metadata["weighted_train_examples"], metadata["train_items"])
        self.assertTrue(train and validation)


if __name__ == "__main__":
    unittest.main()
