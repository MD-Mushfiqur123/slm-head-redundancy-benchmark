"""Figure 2: pruning-ratio vs perplexity Pareto frontier (with variance)."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ._style import apply_style, save


def plot_pareto(
    curves: Dict[str, Dict],
    stem: str,
    outdir: str = "results/figures",
    title: str = "PPL vs. pruning ratio",
    xlabel: str = "Pruning ratio",
    ylabel: str = "WikiText-2 perplexity",
    baselines: Dict[str, float] | None = None,
    annotate_threshold: float | None = None,
):
    """curves: {method: {'ratios': [...], 'ppl': [...], 'ppl_std': [...] }} where
    std is zero for deterministic methods and +-1 sigma for random baselines."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for method, data in curves.items():
        ratios = np.asarray(data["ratios"], dtype=float)
        ppl = np.asarray(data["ppl"], dtype=float)
        std = np.asarray(data.get("ppl_std", np.zeros_like(ppl)), dtype=float)
        ax.plot(ratios, ppl, marker="o", markersize=3.5, linewidth=1.4, label=method)
        if std.max() > 0:
            ax.fill_between(ratios, ppl - std, ppl + std, alpha=0.18)
    if baselines:
        for label, value in baselines.items():
            ax.axhline(value, linestyle=":", linewidth=1, color="gray")
            ax.text(0.6, value, f" {label}", va="bottom", ha="right", fontsize=7, color="gray")
    if annotate_threshold is not None:
        ax.axvline(annotate_threshold, linestyle="--", linewidth=1, color="crimson")
        ax.text(annotate_threshold + 0.005, ax.get_ylim()[0], " 30% threshold",
                fontsize=7, color="crimson", rotation=90, va="bottom")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save(fig, stem, outdir)
