"""Cross-layer global percentile head pruning."""
from __future__ import annotations

import numpy as np

from .base_pruner import BaseHeadPruner, expected_pruned_count


class GlobalHeadPruner(BaseHeadPruner):
    """Flattens all L x Nh head scores, sorts ascending, and removes the
    round(L*Nh*P) globally least-important heads across layer boundaries.

    For grouped-query-attention (GQA) architectures, ``kv_group_size > 1``
    enforces that excision happens at the granularity of whole KV groups
    (complete query groups), which is required for exact weight surgery:
    group importance is the mean of its member query heads."""

    name = "global"

    def __init__(self, num_layers: int, num_heads: int, kv_group_size: int = 1) -> None:
        super().__init__(num_layers, num_heads)
        self.kv_group_size = kv_group_size

    def build_mask(self, importance: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
        imp = np.asarray(importance, dtype=np.float64)
        if imp.shape != (self.num_layers, self.num_heads):
            raise ValueError("importance matrix shape mismatch")
        total = self.num_layers * self.num_heads
        num_pruned = expected_pruned_count(self.num_layers, self.num_heads, ratio)
        mask = np.ones((self.num_layers, self.num_heads), dtype=np.float64)
        if self.kv_group_size <= 1:
            flat_idx = np.argsort(imp.reshape(-1), kind="stable")
            for flat in flat_idx[:num_pruned]:
                mask[flat // self.num_heads, flat % self.num_heads] = 0.0
            return mask
        gs = self.kv_group_size
        if num_pruned % gs != 0:
            raise ValueError(
                f"pruning ratio must excise a multiple of the KV group size {gs}, "
                f"got {num_pruned}"
            )
        groups_flat = []
        for li in range(self.num_layers):
            for g0 in range(0, self.num_heads, gs):
                members = imp[li, g0 : g0 + gs]
                groups_flat.append((float(members.mean()), li, g0))
        groups_flat.sort(key=lambda t: t[0])
        num_groups_pruned = num_pruned // gs
        for _, li, g0 in groups_flat[:num_groups_pruned]:
            mask[li, g0 : g0 + gs] = 0.0
        return mask
