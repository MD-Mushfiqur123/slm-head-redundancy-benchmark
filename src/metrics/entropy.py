"""Shannon attention entropy profiling across decoder layers and heads."""
from __future__ import annotations

from typing import List

import numpy as np
import torch


class EntropyProfiler:
    """Computes mean Shannon entropy of per-head attention distributions,
    ``E_x H(softmax(preye))`` with H(p) = -sum p log2 p, over calibration
    batches using explicit attention outputs."""

    name = "entropy"

    def __init__(self, num_layers: int, num_heads: int) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self._entropy_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
        self._weight = 0.0

    @torch.no_grad()
    def process_batch(self, model, batch) -> None:
        out = model(**batch, output_attentions=True, use_cache=False)
        labels = batch["labels"]
        mask = labels[:, 1:] != -100  # scored positions only
        for li, attn in enumerate(out.attentions):
            a = attn.float()  # [B, Nh, T, T]
            logp = torch.clamp(a, min=1e-12).log2()
            ent = -(a * logp).sum(dim=-1)  # [B, Nh, T]
            w = mask.unsqueeze(1).float()  # [B, 1, T-1] aligned with query positions 0..T-2
            if w.shape[-1] == ent.shape[-1] - 1:
                w = torch.nn.functional.pad(w, (0, 1))
            self._entropy_sum[li] += (ent * w).sum(dim=(0, 2)).cpu().numpy()
        self._weight += float(mask.sum().item())

    def finalize(self) -> np.ndarray:
        if self._weight <= 0:
            raise RuntimeError("Entropy profiler received no calibration data")
        return self._entropy_sum / self._weight
