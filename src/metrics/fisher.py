"""Diagonal empirical Fisher Information importance metric (Theorem 2)."""
from __future__ import annotations

from typing import List

import numpy as np

from .base_metric import BaseImportanceMetric, min_max_normalize


class FisherImportanceScorer(BaseImportanceMetric):
    """Empirical Fisher proxy: mean of squared per-sample directional
    derivatives, ``E[g^2]``, which penalises heads whose sensitivity is
    concentrated on a few idiosyncratic calibration samples."""

    name = "fisher"

    def __init__(self) -> None:
        self._squared_sum = None  # [L, Nh] float64
        self._count = 0
        self._mean_sq: List[np.ndarray] = []

    def process_batch(self, hook_manager, batch) -> None:
        grads = hook_manager.get_per_sample_alpha_grads()  # [L, Nh, B]
        if not isinstance(grads, np.ndarray):
            grads = np.asarray(grads)
        if grads.shape[2] == 0:
            return
        squared = (grads ** 2).mean(axis=2)  # reward expected squared gradient
        if self._squared_sum is None:
            self._squared_sum = squared
        else:
            self._squared_sum = self._squared_sum + squared
        self._mean_sq.append(squared)
        self._count += 1

    def finalize(self, total_tokens: int) -> np.ndarray:
        if self._squared_sum is None or self._count == 0:
            raise RuntimeError("Fisher scorer received no calibration gradients")
        raw = self._squared_sum / self._count
        self.raw_matrix = raw
        self.normalized_matrix = min_max_normalize(raw)
        return raw
