"""First-order Taylor gradient importance metric (Theorem 1)."""
from __future__ import annotations

import numpy as np

from .base_metric import BaseImportanceMetric, min_max_normalize


class TaylorImportanceScorer(BaseImportanceMetric):
    """Token-weighted accumulation of absolute directional derivatives
    ``|g_{l,h}| = |sum_t dL/d alpha_{l,h}|`` across calibration samples."""

    name = "taylor"

    def __init__(self) -> None:
        self._acc = None  # [L, Nh] float64
        self._weight = 0.0

    def process_batch(self, hook_manager, batch) -> None:
        grads = hook_manager.get_per_sample_alpha_grads()  # [L, Nh, B]
        if not isinstance(grads, np.ndarray):
            grads = np.asarray(grads)
        if grads.shape[2] == 0:
            return
        labels = batch["labels"]
        n_per_sample = (labels != -100).sum(dim=1).double().numpy()  # [B]
        b = grads.shape[2]
        weights = n_per_sample[:b]
        weighted = (np.abs(grads) * weights[None, None, :]).sum(axis=2)
        if self._acc is None:
            self._acc = weighted
        else:
            self._acc = self._acc + weighted
        self._weight += float(weights.sum())

    def finalize(self, total_tokens: int) -> np.ndarray:
        if self._acc is None or self._weight <= 0:
            raise RuntimeError("Taylor scorer received no calibration gradients")
        raw = self._acc / self._weight
        self.raw_matrix = raw
        self.normalized_matrix = min_max_normalize(raw)
        return raw
