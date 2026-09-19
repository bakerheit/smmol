import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from github_corpus import decode_blob, decode_source, path_allowed
from training_data import build_splits, read_records, repository_split


class CollectorTests(unittest.TestCase):
    def test_path_filter_keeps_python_and_drops_vendor_or_generated_trees(self):
        self.assertTrue(path_allowed("src/example.py", {".py"}))
        self.assertTrue(path_allowed("tests/test_example.py", {".py"}))
        self.assertFalse(path_allowed("vendor/example.py", {".py"}))
        self.assertFalse(path_allowed("dist/example.py", {".py"}))
        self.assertFalse(path_allowed("src/example.js", {".py"}))

    def test_source_filter_rejects_secrets_generated_files_and_binary_data(self):
        self.assertEqual(decode_source(b"def add(a, b):\n    return a + b\n", 10, 100),
                         "def add(a, b):\n    return a + b\n")
        self.assertIsNone(decode_source(b"# This file is auto-generated; do not edit\nvalue = 1\n", 10, 100))
        self.assertIsNone(decode_source(b"api_key = 'abcdefghijklmnop'\n", 10, 100))
        self.assertIsNone(decode_source(b"code\x00data", 1, 100))

    def test_blob_decoder_checks_encoding(self):
        self.assertEqual(decode_blob({"encoding": "base64", "content": base64.b64encode(b"hi").decode()}), b"hi")
        with self.assertRaisesRegex(RuntimeError, "base64"):
            decode_blob({"encoding": "utf-8", "content": "hi"})


class TrainingDataTests(unittest.TestCase):
    @staticmethod
    def record(identifier, repository, content):
        raw = content.encode()
        return {
            "id": identifier, "repository": repository, "commit": "a" * 40, "license": "MIT",
            "path": identifier + ".py", "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "content": content,
        }

    def test_repository_split_has_no_leakage_and_is_stable(self):
        records = [self.record("a", "one/repo", "a = 1\n"), self.record("b", "two/repo", "b = 2\n"),
                   self.record("c", "three/repo", "c = 3\n")]
        first = repository_split(records, 9)
        second = repository_split(list(reversed(records)), 9)
        self.assertEqual(first, second)
        self.assertFalse(first[0] & first[1])
        self.assertEqual(first[0] | first[1], {"one/repo", "two/repo", "three/repo"})

    def test_record_provenance_is_verified(self):
        item = self.record("a", "one/repo", "a = 1\n")
        other = self.record("b", "two/repo", "b = 2\n")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "records.jsonl"
            path.write_text(json.dumps(item) + "\n" + json.dumps(other) + "\n")
            self.assertEqual(len(read_records(path)), 2)
            item["sha256"] = "0" * 64
            path.write_text(json.dumps(item) + "\n" + json.dumps(other) + "\n")
            with self.assertRaisesRegex(ValueError, "provenance"):
                read_records(path)

    def test_build_splits_adds_replay_only_to_training(self):
        records = [self.record("a", "one/repo", "a = 1\n" * 50),
                   self.record("b", "two/repo", "b = 2\n" * 50)]
        school = [{"word": "cat", "sentence": "The cat naps here."}]
        train, validation, metadata = build_splits(records, school, seed=3, school_replay_fraction=0.1)
        self.assertIn(b"Question: Use the word", train)
        self.assertNotIn(b"Helpful English reply", validation)
        self.assertGreater(metadata["school_replay_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
