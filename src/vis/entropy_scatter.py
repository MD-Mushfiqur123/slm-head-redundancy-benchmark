"""Figure 5: KV-cache memory reduction vs sequence length.

Analytical curves (Theorem 3) plus empirically measured scaling points.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ._style import apply_style, save


def plot_kv_memory(
    seq_lengths: List[int],
    unpruned_mb: List[float],
    pruned_mb: Dict[str, List[float]],
    stem: str = "figure5_kv_cache_memory",
    outdir: str = "results/figures",
    title: str = "KV-cache memory footprint vs. sequence length",
):
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = np.asarray(seq_lengths, dtype=float)
    ax.plot(xs, unpruned_mb, marker="o", markersize=3.5, linewidth=1.5,
            label="Unpruned (analytical)")
    for label, ys in pruned_mb.items():
        ax.plot(xs, ys, marker="s", markersize=3.2, linewidth=1.4, linestyle="--",
                label=f"{label}")
        ax.fill_between(xs, ys, unpruned_mb, alpha=0.12)
    ax.set_xlabel("Sequence length T (tokens)")
    ax.set_ylabel("KV-cache memory (MB), B=8")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save(fig, stem, outdir)
