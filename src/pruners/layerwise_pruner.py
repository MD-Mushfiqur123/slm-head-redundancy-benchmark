"""Uniform per-layer (locally ranked) head pruning."""
from __future__ import annotations

import numpy as np

from .base_pruner import BaseHeadPruner


class LayerwiseHeadPruner(BaseHeadPruner):
    """Ranks heads within each layer independently and removes the
    per-layer fraction P of locally least-important heads (rounded)."""

    name = "layerwise"

    def build_mask(self, importance: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
        imp = np.asarray(importance, dtype=np.float64)
        mask = np.ones((self.num_layers, self.num_heads), dtype=np.float64)
        for li in range(self.num_layers):
            k = int(round(self.num_heads * ratio))
            if k == 0:
                continue
            order = np.argsort(imp[li], kind="stable")
            mask[li, order[:k]] = 0.0
        return mask
