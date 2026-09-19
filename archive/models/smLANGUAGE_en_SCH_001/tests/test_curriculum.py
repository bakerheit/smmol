import unittest

from curriculum import LEVELS, corpus_problems, item_problems


class CurriculumTests(unittest.TestCase):
    def test_level_counts_grow_by_one_hundred(self):
        self.assertEqual(len(LEVELS), 14)
        self.assertEqual([level["count"] for level in LEVELS], list(range(100, 1500, 100)))
        self.assertEqual(sum(level["count"] for level in LEVELS), 10_500)

    def test_valid_item_needs_target_as_a_whole_word(self):
        good = {"level": "preschool", "word": "cat", "sentence": "The cat naps here."}
        bad = {"level": "preschool", "word": "cat", "sentence": "I see cattle nearby."}
        self.assertEqual(item_problems(good), [])
        self.assertIn("sentence does not contain the target word", item_problems(bad))

    def test_duplicates_cross_level_are_rejected(self):
        items = [
            {"level": "preschool", "word": "apple", "sentence": "I eat one apple."},
            {"level": "kindergarten", "word": "apple", "sentence": "The red apple fell down."},
        ]
        self.assertIn("duplicate target word: apple", corpus_problems(items))

    def test_later_levels_reject_grammar_glue_as_targets(self):
        item = {"level": "kindergarten", "word": "and", "sentence": "Tom and Ana play outside."}
        self.assertIn("target word is grammar glue, not curriculum vocabulary", item_problems(item))


if __name__ == "__main__":
    unittest.main()
