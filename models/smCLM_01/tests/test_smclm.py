"""smCLM_01: the made-up world keeps its promises, and the scoring and models behave.

    python3 -m unittest discover -s tests
"""
import os
import random
import sys
import unittest
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402
from model import ConceptModel, WordModel  # noqa: E402
from world import World, tokens  # noqa: E402


class WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World()

    def test_sentences_only_use_words_whose_ideas_fit(self):
        rng = random.Random(0)
        for _ in range(5000):
            words, filled = self.world.sentence(rng, detail=True)
            for word, need, avoid in filled:
                self.assertTrue(need <= self.world.tags[word], (" ".join(words), word, need))
                self.assertFalse(avoid & self.world.tags[word], (" ".join(words), word, avoid))

    def test_every_idea_is_shown_by_some_sentence(self):
        shown = set().union(*[p[2] for t in self.world.templates for p in t if p[0] == "slot"])
        self.assertEqual(shown, set(self.world.concepts))

    def test_held_out_words_get_used(self):
        counts = Counter(t for s in self.world.corpus(20000, 1) for t in s)
        for word in self.world.held_out:
            self.assertGreater(counts[word], 30, word)

    def test_teach_sentences_only_use_known_words(self):
        known = set(self.world.frames) | set(self.world.nouns)
        for word, sentences in self.world.teach.items():
            for s in sentences:
                toks = tokens(s)
                self.assertIn(word, toks)
                self.assertEqual([t for t in toks if t != word and t not in known], [], s)


class ModelTests(unittest.TestCase):
    def test_scoring(self):
        tags = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 0]])
        held = torch.tensor([True, True])
        self.assertEqual(train.score(tags, tags, held), (1.0, 1.0))
        backwards = 1 - tags
        self.assertEqual(train.score(backwards, tags, held)[0], 0.0)

    def test_readers_shapes(self):
        b, t, k = 4, 12, 51
        frames = torch.randint(3, 20, (b, t))
        pad = torch.zeros(b, t, dtype=torch.bool)
        is_noun = torch.zeros(b, t, dtype=torch.bool)
        is_noun[:, 1] = True
        target = torch.full((b,), 3)
        ideas = torch.rand(b, t, k)
        self.assertEqual(ConceptModel(30, k)(frames, ideas, is_noun, target, pad).shape, (b, k))
        nouns = torch.randint(0, 100, (b, t))
        self.assertEqual(WordModel(30, 100)(frames, nouns, is_noun, target, pad).shape, (b, 100))

    def test_masking_hides_the_target(self):
        world = World()
        data = train.encode(world, [["the", "driver", "steers", "the", "car"]], 12)
        rows, cols = torch.tensor([0]), torch.tensor([4])
        f, n, is_noun, p, target, x = train.masked(data, rows, cols, torch.ones(len(world.nouns), len(world.concepts)))
        self.assertEqual(world.nouns[target[0]], "car")
        self.assertFalse(is_noun[0, 4])
        self.assertEqual(x[0, 4].sum().item(), 0)  # the hidden word's ideas don't leak in


if __name__ == "__main__":
    unittest.main()
