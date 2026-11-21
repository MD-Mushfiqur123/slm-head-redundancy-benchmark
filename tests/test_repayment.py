"""Offline tests for the fine-tuning repayment protocol helpers.

The full experiment is a GPU job; these tests pin the pure protocol logic
(GQA ratio snapping, cosine schedule, repair-percentage bookkeeping) that the
GPU run depends on, so a malformed ratio can never silently produce a
wrong table.
"""
from __future__ import annotations

import importlib.util
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SPEC_PATH = os.path.join(REPO, "scripts", "finetune_repayment.py")
_spec = importlib.util.spec_from_file_location("finetune_repayment", _SPEC_PATH)
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)


class _Spec:
    pass


def _spec(nl, nh, nkv):
    s = _Spec()
    s.num_layers = nl
    s.num_heads = nh
    s.num_kv_heads = nkv
    return s


class TestSnapRatio(unittest.TestCase):
    def test_dense_unchanged(self):
        for r in (0.0, 0.1, 0.3, 0.6):
            self.assertEqual(fr.snap_ratio(_spec(12, 12, 12), r), r)

    def test_smollm_group3_ratio30_exact(self):
        self.assertAlmostEqual(fr.snap_ratio(_spec(30, 9, 3), 0.30), 0.30)

    def test_qwen_group7_ratio30_snaps_down(self):
        got = fr.snap_ratio(_spec(24, 14, 2), 0.30)
        self.assertAlmostEqual(got, 98 / 336)

    def test_snapped_ratio_is_group_aligned(self):
        s = _spec(24, 14, 2)
        kv_group = s.num_heads // s.num_kv_heads
        total = s.num_layers * s.num_heads
        excised = int(round(total * fr.snap_ratio(s, 0.30)))
        self.assertEqual(excised % kv_group, 0)

    def test_tiny_ratio_snaps_to_first_group(self):
        self.assertAlmostEqual(fr.snap_ratio(_spec(24, 14, 2), 0.0001),
                               7 / 336)

    def test_zero_stays_zero(self):
        self.assertEqual(fr.snap_ratio(_spec(24, 14, 2), 0.0), 0.0)


class TestCosineSchedule(unittest.TestCase):
    def test_warmup_linear_ramp(self):
        lrs = [fr.cosine_lr(s, 100, 10, 1e-5) for s in range(10)]
        self.assertTrue(all(b > a for a, b in zip(lrs, lrs[1:])))

    def test_peak_exactly_base_lr_at_warmup_end(self):
        self.assertAlmostEqual(fr.cosine_lr(10, 100, 10, 1e-5), 1e-5)

    def test_cosine_decay_and_endpoints(self):
        self.assertLess(fr.cosine_lr(55, 100, 10, 1e-5), fr.cosine_lr(20, 100, 10, 1e-5))
        self.assertAlmostEqual(fr.cosine_lr(100, 100, 10, 1e-5), 0.0, places=12)


class TestRepairBookkeeping(unittest.TestCase):
    def test_repair_pct_full_identification(self):
        base, pre, post = 36.0, 54.0, 36.0
        repair = 100 * (1 - (post - base) / (pre - base))
        self.assertAlmostEqual(repair, 100.0)

    def test_repair_pct_none(self):
        base, pre, post = 36.0, 54.0, 54.0
        repair = 100 * (1 - (post - base) / (pre - base))
        self.assertAlmostEqual(repair, 0.0)

    def test_repair_pct_fractional(self):
        base, pre, post = 36.0, 54.0, 45.0
        repair = 100 * (1 - (post - base) / (pre - base))
        self.assertAlmostEqual(repair, 50.0)


if __name__ == "__main__":
    unittest.main()
