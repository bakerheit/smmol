"""Phase 2 scoring tests: the arm reads the right byte, the episodes really are paired, and a
frozen evaluation that touches a parameter is a hard error rather than a silent result."""

from __future__ import annotations

import math
import unittest

import torch

import evaluate
import train_recall
import world
from cells import RTSModel


def tiny_model(cell: str = "gated", seed: int = 3) -> RTSModel:
    torch.manual_seed(seed)
    return RTSModel(cell, dim=32, layers=1, heads=2, key_dim=4, value_dim=4)


class ScoringPosition(unittest.TestCase):
    def test_batched_and_sequential_agree(self):
        """The fast padded path and the one-at-a-time path must read the same position."""
        model = tiny_model()
        batch = world.episodes(12, pairs=4, gap=32, seed=world.SELECT_SEED)
        fast = evaluate.answer_logprobs(model, batch, chunk=5)
        slow = evaluate.answer_logprobs_sequential(model, batch, [True] * len(batch))
        for a, b in zip(fast, slow):
            self.assertTrue(torch.allclose(a, b, atol=1e-5), "padded and sequential disagree")

    def test_padding_cannot_leak(self):
        """Ragged rows in one chunk score identically to each row scored alone."""
        model = tiny_model("fast")
        batch = world.episodes(4, pairs=2, gap=8, seed=5) + \
            world.episodes(4, pairs=16, gap=128, seed=5)
        together = evaluate.answer_logprobs(model, batch, chunk=len(batch))
        for index, episode in enumerate(batch):
            alone = evaluate.answer_logprobs(model, [episode], chunk=1)[0]
            self.assertTrue(torch.allclose(together[index], alone, atol=1e-5))

    def test_arm_reads_the_answer_slot(self):
        """A model forced to copy its input byte should predict `data[answer_index]` exactly."""
        episode = world.sample_episode(world.derive_seed(1), pairs=4, gap=32)
        inputs, targets, mask, _ = world.batch_tensors([episode])
        self.assertEqual(int(mask.sum()), 1)
        position = int(mask[0].nonzero()[0])
        self.assertEqual(position, episode.answer_index - 1)
        self.assertEqual(int(inputs[0, position]), episode.data[episode.answer_index - 1])
        self.assertEqual(int(targets[0, position]), episode.answer)
        self.assertEqual(episode.data[episode.answer_index], episode.answer)
        # The byte before the answer is the queried key, so the scored slot really is the recall.
        self.assertEqual(int(inputs[0, position]), episode.query_key)

    def test_logprobs_are_normalised(self):
        model = tiny_model()
        rows = evaluate.answer_logprobs(model, world.episodes(6, 4, 32, seed=7))
        for row in rows:
            self.assertEqual(row.numel(), 256)
            self.assertAlmostEqual(float(row.exp().sum()), 1.0, places=4)


class FrozenEvaluation(unittest.TestCase):
    def test_evaluate_plan_does_not_move_parameters(self):
        model = tiny_model()
        before = world.parameter_fingerprint(model)
        plan = world.EvalPlan(pairs=4, gap=32, trials=25, seed=world.EVAL_SEED)
        evaluate.evaluate_plan(model, plan)
        self.assertEqual(before, world.parameter_fingerprint(model))

    def test_a_mutating_arm_is_an_error(self):
        model = tiny_model()

        def bad_arm(episode, reset):
            with torch.no_grad():
                model.decoder.weight.add_(1e-3)
            return episode.answer

        plan = world.EvalPlan(pairs=4, gap=32, trials=5, seed=world.EVAL_SEED)
        with self.assertRaises(RuntimeError):
            world.run_plan(plan, {"bad": bad_arm}, parameter_modules={"bad": model})

    def test_grad_is_not_enabled_inside_scoring(self):
        model = tiny_model()
        rows = evaluate.answer_logprobs(model, world.episodes(4, 4, 32, seed=9))
        for row in rows:
            self.assertFalse(row.requires_grad)
            self.assertIsNone(row.grad_fn)


class Pairing(unittest.TestCase):
    def test_model_and_lazy_arms_share_one_episode_hash(self):
        model = tiny_model()
        plan = world.EvalPlan(pairs=8, gap=32, trials=40, seed=world.EVAL_SEED)
        summaries = evaluate.evaluate_plan(model, plan)
        hashes = {row["episode_set_hash"] for row in summaries.values()}
        self.assertEqual(len(hashes), 1, "arms were not paired")
        self.assertIn("model", summaries)
        self.assertIn("most_common", summaries)

    def test_selection_and_report_sets_are_disjoint(self):
        """Nothing selected on SELECT_SEED may be scored on EVAL_SEED."""
        self.assertNotEqual(world.SELECT_SEED, world.EVAL_SEED)
        self.assertNotEqual(world.TRAIN_SEED, world.EVAL_SEED)
        self.assertNotEqual(world.TRAIN_SEED, world.SELECT_SEED)
        report = world.episodes(200, 16, 128, seed=world.EVAL_SEED)
        select = world.episodes(200, 16, 128, seed=world.SELECT_SEED)
        self.assertEqual(len(set(e.data for e in report) & set(e.data for e in select)), 0)

    def test_training_stream_never_hits_the_reported_episodes(self):
        reported = set(e.data for e in world.episodes(300, 16, 128, seed=world.EVAL_SEED))
        stream = train_recall.EpisodeStream(world.TRAIN_SEED, 0, (16,), (128,))
        produced = bytearray()
        while stream.episodes_made < 300:
            produced += stream.take(64)
        blob = bytes(produced)
        for episode in reported:
            self.assertNotIn(episode, blob)


class Metrics(unittest.TestCase):
    def test_rank_and_nll_match_a_hand_computation(self):
        logprobs = torch.full((256,), -20.0)
        logprobs[65] = math.log(0.5)
        logprobs[66] = math.log(0.3)
        logprobs[67] = math.log(0.2)
        correct, nll, rank = world._score(logprobs, 66)
        self.assertFalse(correct)
        self.assertAlmostEqual(nll, -math.log(0.3), places=6)
        self.assertEqual(rank, 2)
        correct, nll, rank = world._score(logprobs, 65)
        self.assertTrue(correct)
        self.assertEqual(rank, 1)

    def test_untrained_model_sits_near_chance(self):
        model = tiny_model()
        score = evaluate.selection_score(model, pairs=16, gap=128, trials=200)
        self.assertLess(score["accuracy"], 0.20)
        self.assertGreater(score["nll"], 3.0)

    def test_futility_reference_floors_are_ordered(self):
        reference = evaluate.futility_reference(16, 32, trials=200)
        self.assertAlmostEqual(reference["alphabet_nll"], math.log(33), places=6)
        self.assertLess(reference["in_episode_nll"], reference["alphabet_nll"])

    def test_selection_score_is_deterministic(self):
        model = tiny_model()
        first = evaluate.selection_score(model, trials=100)
        second = evaluate.selection_score(model, trials=100)
        self.assertEqual(first["episode_set_hash"], second["episode_set_hash"])
        self.assertAlmostEqual(first["nll"], second["nll"], places=6)


class TrainingStream(unittest.TestCase):
    def test_streams_are_independent_and_reproducible(self):
        a = train_recall.StreamBatch(4, world.TRAIN_SEED, (4,), (32,))
        b = train_recall.StreamBatch(4, world.TRAIN_SEED, (4,), (32,))
        first, second = a.window(64), b.window(64)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first[0], first[1]), "streams are not independent")
        self.assertEqual(first.shape, (4, 65))

    def test_window_accounting(self):
        batch = train_recall.StreamBatch(3, world.TRAIN_SEED, (8,), (32, 128))
        batch.window(100)
        batch.window(100)
        self.assertEqual(batch.bytes_served, 600)

    def test_training_mix_is_the_registered_one(self):
        self.assertEqual(train_recall.TRAIN_GAPS, (32, 128))
        self.assertNotIn(8, train_recall.TRAIN_GAPS)
        self.assertNotIn(512, train_recall.TRAIN_GAPS)

    def test_report_cells_cover_both_axes(self):
        cells = train_recall.report_cells()
        self.assertIn((16, 128), cells)
        for gap in world.GAPS:
            self.assertIn((16, gap), cells)
        for pairs in world.PAIR_COUNTS:
            self.assertIn((pairs, 128), cells)
        self.assertEqual(len(cells), len(set(cells)))


class Guards(unittest.TestCase):
    def test_controls_are_refused_online(self):
        for cell in ("delta", "gru"):
            model = RTSModel(cell, dim=16, layers=1, heads=2, key_dim=4, value_dim=4)
            self.assertFalse(model.supports_online)

    def test_frozen_decay_is_exactly_the_requested_half_life(self):
        class Args:
            seed, dim, layers, heads, key_dim, value_dim = 1, 32, 1, 2, 4, 4
            freeze_decay = 128.0

        model = train_recall.build_model("leaky", Args())
        for module in model.cells:
            self.assertFalse(module.decay_logit.requires_grad)
            half = evaluate.decay_half_life(module.decay_logit.detach())
            self.assertTrue(torch.allclose(half, torch.full_like(half, 128.0), atol=1e-3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
