"""Structural magnitude baselines: L1/L2 weight-norm importance scores."""
from __future__ import annotations

import numpy as np
import torch

from ..core.model_loader import locate_attentions


class WeightNormScorer:
    """Per-head structural weight norms computed directly from QKV/output
    projection parameters — calibration-free heuristic baseline."""

    name = "weight_norm_l2"

    def __init__(self, order: int = 2) -> None:
        self.order = order

    def compute(self, model) -> np.ndarray:
        slicing = locate_attentions(model)
        l, nh = len(slicing), slicing[0].num_heads
        matrix = np.zeros((l, nh), dtype=np.float64)
        for li, sl in enumerate(slicing):
            for h in range(nh):
                total = 0.0
                if sl.is_conv1d:
                    w = sl.qkv_param.weight.detach().float()  # [d, 3d]
                    for key in ("q", "k", "v"):
                        total += float(
                            torch.linalg.vector_norm(w[:, sl.layout[h][key]], ord=self.order)
                            ** self.order
                        )
                    wo = sl.o_proj.weight.detach().float()  # [768 in, 768 out]
                    total += float(
                        torch.linalg.vector_norm(
                            wo[sl.layout[h]["v"], :],  # rows = input features per head
                            ord=self.order,
                        )
                        ** self.order
                    )
                else:
                    q, k, v = (
                        sl.qkv_param["q"].weight.detach().float(),
                        sl.qkv_param["k"].weight.detach().float(),
                        sl.qkv_param["v"].weight.detach().float(),
                    )
                    total += float(
                        torch.linalg.vector_norm(q[sl.layout[h]["q"], :], ord=self.order)
                        ** self.order
                    )
                    kv_head, cols = sl.layout[h]["k"]
                    dk = sl.d_head
                    total += float(
                        torch.linalg.vector_norm(
                            k[kv_head * dk : (kv_head + 1) * dk, :], ord=self.order
                        )
                        ** self.order
                    )
                    total += float(
                        torch.linalg.vector_norm(
                            v[kv_head * dk : (kv_head + 1) * dk, :], ord=self.order
                        )
                        ** self.order
                    )
                    wo = sl.o_proj.weight.detach().float()  # [out, in]
                    total += float(
                        torch.linalg.vector_norm(
                            wo[:, slice(h * sl.d_head, (h + 1) * sl.d_head)],
                            ord=self.order,
                        )
                        ** self.order
                    )
                matrix[li, h] = total ** (1.0 / self.order)
        return matrix
