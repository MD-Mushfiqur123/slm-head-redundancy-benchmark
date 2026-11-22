"""Abstract base class for head importance metrics.

A metric consumes gradient / activation telemetry captured by the
:class:`src.core.hooks.HeadGatingHookManager` over calibration batches and
produces an importance matrix of shape ``[num_layers, num_heads]``.
"""
from __future__ import annotations

import abc
from typing import Iterable

import numpy as np


def min_max_normalize(matrix: np.ndarray) -> np.ndarray:
    """Map an importance matrix to [0, 1] with min-max scaling per matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-30:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


class BaseImportanceMetric(abc.ABC):
    """Interface implemented by every head-importance metric."""

    #: short identifier used in CSV columns and figure legends
    name: str = "base"

    #: defensive-access targets: None until :meth:`finalize` stores the
    #: accumulated matrix, so callers may probe ``scorer.raw_matrix`` early
    raw_matrix = None
    normalized_matrix = None

    @abc.abstractmethod
    def process_batch(self, hook_manager, batch) -> None:
        """Accumulate telemetry for one calibration batch (already run
        through forward + backward by the caller)."""

    @abc.abstractmethod
    def finalize(self, total_tokens: int) -> np.ndarray:
        """Fold accumulated telemetry into an [L, Nh] importance matrix."""

    def compute(self, model, hook_manager, batches, tokenizer=None) -> np.ndarray:
        """Full calibration loop: forward + backward over all batches."""
        total_tokens = 0
        for batch in batches:
            hook_manager.reset_gradients()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            n_scored = int((batch["labels"] != -100).sum().item())
            total_tokens += n_scored
            self.process_batch(hook_manager, batch)
        return self.finalize(total_tokens)
