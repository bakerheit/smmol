"""Focused checks for the local Muon experiment implementation."""

import unittest

import torch

from muon import SingleDeviceMuon, adjustment, zeropower_newton_schulz


class MuonTests(unittest.TestCase):
    def test_zeropower_preserves_shape_and_is_finite(self):
        for shape in ((16, 8), (8, 16), (8, 8)):
            value = torch.randn(*shape)
            result = zeropower_newton_schulz(value)
            self.assertEqual(result.shape, value.shape)
            self.assertTrue(torch.isfinite(result).all())

    def test_adjustments_match_published_formulas(self):
        self.assertAlmostEqual(adjustment((192, 576), "original"), 1.0)
        self.assertAlmostEqual(adjustment((576, 192), "original"), 3 ** 0.5)
        self.assertAlmostEqual(adjustment((192, 576), "match_rms_adamw"), 0.2 * 576 ** 0.5)

    def test_step_changes_a_matrix(self):
        parameter = torch.nn.Parameter(torch.randn(16, 8))
        before = parameter.detach().clone()
        parameter.grad = torch.randn_like(parameter)
        optimizer = SingleDeviceMuon([parameter], lr=0.01)
        optimizer.step()
        self.assertFalse(torch.equal(parameter, before))
        self.assertTrue(torch.isfinite(parameter).all())

    def test_rejects_vector_parameters(self):
        with self.assertRaises(ValueError):
            SingleDeviceMuon([torch.nn.Parameter(torch.randn(8))], lr=0.01)


if __name__ == "__main__":
    unittest.main()
