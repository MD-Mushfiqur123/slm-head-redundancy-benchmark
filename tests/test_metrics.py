"""Metric correctness: non-negativity, scaling and normalisation."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.base_metric import min_max_normalize
from src.metrics.fisher import FisherImportanceScorer
from src.metrics.taylor import TaylorImportanceScorer
from src.metrics.weight_norm import WeightNormScorer
from tests.test_pruners import _toy_llama
from src.core.model_loader import locate_attentions


class _FakeHM:
    def __init__(self, values, batch=4):
        self._v = values  # [L, Nh]
        self._b = batch

    def get_per_sample_alpha_grads(self):
        return self._v[:, :, None] * np.arange(1, self._b + 1)[None, None, :]

    def reset_gradients(self):
        pass


class FakeBatch(dict):
    def __init__(self, tokens_per_sample=(6, 5, 7, 4)):
        super().__init__()
        import torch

        labels = torch.ones((4, 8), dtype=torch.long) * -100
        for i, n in enumerate(tokens_per_sample):
            labels[i, 1 : 1 + n] = 1
        self["labels"] = labels


class TestTaylor(unittest.TestCase):
    def test_non_negative_and_normalization(self):
        scorer = TaylorImportanceScorer()
        vals = np.random.RandomState(0).randn(3, 4)
        hm = _FakeHM(vals)
        for _ in range(3):
            scorer.process_batch(hm, FakeBatch())
        matrix = scorer.finalize(total_tokens=66)
        self.assertEqual(matrix.shape, (3, 4))
        self.assertGreaterEqual(float(matrix.min()), 0.0)
        self.assertGreaterEqual(float(matrix.max()), 0.0)

    def test_zero_importance_head_is_reported(self):
        scorer = TaylorImportanceScorer()
        vals = np.zeros((2, 2))
        vals[0, 0] = 3.0
        hm = _FakeHM(vals)
        scorer.process_batch(hm, FakeBatch())
        matrix = scorer.finalize(total_tokens=22)
        # token-weighted |g| = 3 * sum(w_s * s)/sum(w_s) with scale [1,2,3,4]
        expected = 3.0 * (6 * 1 + 5 * 2 + 7 * 3 + 4 * 4) / 22.0
        self.assertAlmostEqual(float(matrix[0, 0]), expected, places=12)
        self.assertAlmostEqual(float(matrix[1, 1]), 0.0, places=12)
        self.assertAlmostEqual(float(scorer.normalized_matrix[0, 0]), 1.0, places=12)


class TestFisher(unittest.TestCase):
    def test_non_negative_and_variance_sensitivity(self):
        scorer = FisherImportanceScorer()
        vals = np.random.RandomState(0).randn(3, 4)
        hm = _FakeHM(vals)
        for _ in range(3):
            scorer.process_batch(hm, FakeBatch())
        matrix = scorer.finalize(total_tokens=66)
        self.assertEqual(matrix.shape, (3, 4))
        self.assertGreaterEqual(float(matrix.min()), 0.0)

    def test_constant_positive_gradient_scores_low_variance(self):
        s1 = FisherImportanceScorer()
        hm1 = _FakeHM(np.ones((1, 1)))
        # _FakeHM scales by sample index; emulate identical per-sample values
        class ConstHM:
            def get_per_sample_alpha_grads(self):
                return np.ones((1, 1, 4))

            def reset_gradients(self):
                pass

        for _ in range(3):
            s1.process_batch(ConstHM(), FakeBatch())
        raw = s1.finalize(10)
        # E[g^2] = 1 for constant gradient of 1
        self.assertAlmostEqual(float(raw[0, 0]), 1.0, places=9)


class TestWeightNorm(unittest.TestCase):
    def test_l2_norm_shapes_and_ordering(self):
        import torch

        model, _ = _toy_llama()
        scorer = WeightNormScorer(order=2)
        matrix = scorer.compute(model)
        self.assertEqual(matrix.shape, (2, 6))
        self.assertTrue(bool((matrix > 0).all()))

    def test_l1_alias(self):
        import torch

        model, _ = _toy_llama()
        _ = WeightNormScorer(order=1).compute(model)


if __name__ == "__main__":
    unittest.main()
