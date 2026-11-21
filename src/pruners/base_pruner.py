"""Abstract head pruner with exact-mask validation utilities."""
from __future__ import annotations

import abc

import numpy as np


def expected_pruned_count(num_layers: int, num_heads: int, ratio: float) -> int:
    """round(L * Nh * P) — the exact number of heads a mask must contain."""
    return int(round(num_layers * num_heads * ratio))


def validate_mask(mask: np.ndarray, num_layers: int, num_heads: int, ratio: float) -> None:
    arr = np.asarray(mask)
    if arr.shape != (num_layers, num_heads):
        raise ValueError(f"mask shape {arr.shape} != {(num_layers, num_heads)}")
    unique = set(np.unique(arr).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"mask must be binary, found values {unique}")
    pruned = int((arr == 0).sum())
    expected = expected_pruned_count(num_layers, num_heads, ratio)
    if pruned != expected:
        raise ValueError(f"mask has {pruned} pruned heads, expected {expected}")


class BaseHeadPruner(abc.ABC):
    """Interface: produce a binary head mask from an importance matrix."""

    name: str = "base"

    def __init__(self, num_layers: int, num_heads: int) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads

    @abc.abstractmethod
    def build_mask(self, importance: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
        """Return a binary [L, Nh] mask with round(L*Nh*ratio) zeros, where
        zeros mark heads selected for excision (lowest importance)."""

    def mask_summary(self, mask: np.ndarray) -> dict:
        arr = np.asarray(mask)
        return {
            "active_heads": int(arr.sum()),
            "pruned_heads": int((arr == 0).sum()),
            "survival_rate": float(arr.mean()),
        }
