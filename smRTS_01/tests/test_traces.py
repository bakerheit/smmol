"""Phase 0 gates for smRTS_01: exact one-layer traces, explicit depth bias, and no hidden graph."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import unittest
import unittest.mock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import online  # noqa: E402
import tbptt  # noqa: E402
from cells import RTSModel, decay_half_life, spread_decay_logits  # noqa: E402


def fixture(kind: str, layers: int = 1) -> RTSModel:
    torch.manual_seed(1701)
    model = RTSModel(kind, dim=6, layers=layers, heads=2, key_dim=3, value_dim=2).double()
    model.set_decay(0.99)
    return model


def bytes_for(length: int, batch: int = 1):
    generator = torch.Generator().manual_seed(901)
    inputs = torch.randint(0, 256, (batch, length), generator=generator)
    targets = torch.randint(0, 256, (batch, length), generator=generator)
    return inputs, targets


def compare(kind: str, length: int, layers: int = 1):
    base = fixture(kind, layers)
    traced, exact = copy.deepcopy(base), copy.deepcopy(base)
    inputs, targets = bytes_for(length)
    online_loss, online_logits, state = online.sequence_gradients(traced, inputs, targets)
    bptt_loss, bptt_logits = tbptt.sequence_gradients(exact, inputs, targets)
    rows = {}
    for (name, got), (other, want) in zip(traced.named_parameters(), exact.named_parameters()):
        assert name == other
        difference = (got.grad - want.grad).norm()
        denominator = want.grad.norm().clamp_min(1e-15)
        cosine = F.cosine_similarity(got.grad.flatten(), want.grad.flatten(), dim=0).item()
        rows[name] = {"relative": (difference / denominator).item(), "cosine": cosine}
    return online_loss, bptt_loss, online_logits, bptt_logits, rows, state


class Decays(unittest.TestCase):
    def test_spread_covers_one_to_1024_steps(self):
        logits = spread_decay_logits(9, 1.0, 1024.0, dtype=torch.float64)
        half_lives = decay_half_life(logits)
        self.assertAlmostEqual(half_lives[0].item(), 1.0, places=10)
        self.assertAlmostEqual(half_lives[-1].item(), 1024.0, places=8)
        self.assertTrue(torch.all(half_lives[1:] > half_lives[:-1]))


class ExactOneLayer(unittest.TestCase):
    def assert_exact(self, kind: str, length: int):
        online_loss, bptt_loss, got_logits, want_logits, rows, state = compare(kind, length)
        self.assertAlmostEqual(online_loss, bptt_loss, places=10)
        torch.testing.assert_close(got_logits, want_logits, rtol=1e-12, atol=1e-12)
        for name, result in rows.items():
            self.assertLess(result["relative"], 1e-6, "%s %s" % (name, result))
            self.assertAlmostEqual(result["cosine"], 1.0, places=10, msg=name)
        self.assertTrue(online.graph_is_severed(state))

    def test_32_byte_gradient_gate(self):
        for kind in ("leaky", "gated", "fast"):
            with self.subTest(cell=kind):
                self.assert_exact(kind, 32)

    def test_128_byte_carry_gate(self):
        for kind in ("leaky", "gated", "fast"):
            with self.subTest(cell=kind):
                self.assert_exact(kind, 128)

    def test_a_deliberate_no_carry_mutant_fails(self):
        model = fixture("leaky")
        exact = copy.deepcopy(model)
        inputs, targets = bytes_for(32)
        original = online._leaky_step

        def no_carry(cell, x, previous, traces, bridge_input, byte_ids):
            for trace in traces.values():
                trace.zero_()
            return original(cell, x, previous, traces, bridge_input, byte_ids)

        with unittest.mock.patch.object(online, "_leaky_step", no_carry):
            online.sequence_gradients(model, inputs, targets)
        tbptt.sequence_gradients(exact, inputs, targets)
        got = model.cells[0].decay_logit.grad
        want = exact.cells[0].decay_logit.grad
        relative = ((got - want).norm() / want.norm()).item()
        self.assertGreater(relative, 0.1, "the gate did not notice a missing trace carry")

    def test_traced_parameters_are_not_double_counted(self):
        traced_names = {
            "leaky": ("embedding.weight", "cells.0.decay_logit"),
            "gated": ("cells.0.decay_logit", "cells.0.value.weight", "cells.0.gate.weight"),
            "fast": ("cells.0.decay_logit", "cells.0.key.weight", "cells.0.value.weight"),
        }
        inputs, targets = bytes_for(32)
        for kind, names in traced_names.items():
            with self.subTest(cell=kind):
                traced = fixture(kind)
                exact = copy.deepcopy(traced)
                online.sequence_gradients(traced, inputs, targets)
                tbptt.sequence_gradients(exact, inputs, targets)
                traced_parameters = dict(traced.named_parameters())
                exact_parameters = dict(exact.named_parameters())
                for name in names:
                    ratio = (traced_parameters[name].grad.norm()
                             / exact_parameters[name].grad.norm().clamp_min(1e-15)).item()
                    self.assertAlmostEqual(ratio, 1.0, places=10, msg=name)


class DepthApproximation(unittest.TestCase):
    def test_two_layers_are_reported_as_approximate_not_exact(self):
        _, _, got_logits, want_logits, rows, state = compare("gated", 32, layers=2)
        torch.testing.assert_close(got_logits, want_logits, rtol=1e-12, atol=1e-12)
        self.assertTrue(online.graph_is_severed(state))
        self.assertGreater(max(row["relative"] for row in rows.values()), 1e-5)


class StateAndTrainerContracts(unittest.TestCase):
    def test_first_token_logits_and_loss_match_tbptt(self):
        for kind in ("leaky", "gated", "fast"):
            with self.subTest(cell=kind):
                model = fixture(kind)
                reference = copy.deepcopy(model)
                inputs, targets = bytes_for(1, batch=3)
                state = online.initialize(model, 3)
                got_loss, got_logits, state = online.step(
                    model, inputs[:, 0], targets[:, 0], state)
                want_loss, want_logits, _ = tbptt.sequence_loss(reference, inputs, targets)
                self.assertAlmostEqual(got_loss, want_loss.item(), places=12)
                torch.testing.assert_close(got_logits, want_logits[:, 0], rtol=1e-12, atol=1e-12)
                self.assertTrue(online.graph_is_severed(state))

    def test_online_batched_stream_zero_matches_batch_one_and_tbptt(self):
        for kind in ("leaky", "gated", "fast"):
            with self.subTest(cell=kind):
                single = fixture(kind)
                batched = copy.deepcopy(single)
                reference = copy.deepcopy(single)
                inputs, targets = bytes_for(1, batch=3)
                _, single_logits, _ = online.step(
                    single, inputs[:1, 0], targets[:1, 0], online.initialize(single, 1))
                _, batched_logits, _ = online.step(
                    batched, inputs[:, 0], targets[:, 0], online.initialize(batched, 3))
                _, reference_logits, _ = tbptt.sequence_loss(reference, inputs[:1], targets[:1])
                torch.testing.assert_close(batched_logits[0], single_logits[0], rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(single_logits[0], reference_logits[0, 0], rtol=1e-12, atol=1e-12)
                single_loss = F.cross_entropy(single_logits, targets[:1, 0])
                reference_loss = F.cross_entropy(reference_logits[:, 0], targets[:1, 0])
                self.assertAlmostEqual(single_loss.item(), reference_loss.item(), places=12)

    def test_trace_storage_scales_with_streams_and_contains_no_graph(self):
        for kind in ("leaky", "gated", "fast"):
            one = online.initialize(fixture(kind), 1)
            three = online.initialize(fixture(kind), 3)
            self.assertEqual(online.trace_bytes(three), 3 * online.trace_bytes(one))
            self.assertTrue(online.graph_is_severed(one))
            self.assertTrue(online.graph_is_severed(three))

    def test_300_online_steps_do_not_retain_a_graph(self):
        model = fixture("fast")
        state = online.initialize(model, 1)
        byte, target = torch.tensor([4]), torch.tensor([9])
        for _ in range(300):
            model.zero_grad(set_to_none=True)
            _, _, state = online.step(model, byte, target, state)
        self.assertTrue(online.graph_is_severed(state))

    def test_delta_and_gru_are_tbptt_controls_only(self):
        inputs, targets = bytes_for(5, batch=2)
        for kind in ("delta", "gru"):
            with self.subTest(cell=kind):
                model = fixture(kind)
                self.assertFalse(model.supports_online)
                loss, logits, states = tbptt.sequence_loss(model, inputs, targets)
                loss.backward()
                self.assertEqual(logits.shape, (2, 5, 256))
                self.assertEqual(len(states), 1)
                self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
                with self.assertRaisesRegex(ValueError, "no proved online trace"):
                    online.initialize(model, 1)


if __name__ == "__main__":
    unittest.main()
