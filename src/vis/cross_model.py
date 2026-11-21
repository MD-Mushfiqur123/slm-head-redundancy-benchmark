"""Figure 6: cross-model sparsity retention (comparative Pareto curves)."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ._style import apply_style, save


def plot_retention_curves(
    curves: Dict[str, Dict],
    stem: str = "figure6_cross_model_retention",
    outdir: str = "results/figures",
    title: str = "Cross-model sparsity retention curves",
):
    """curves: {model_label: {'ratios': [...], 'ppl': [...], 'ppl_delta_pct': [...]}}."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for label, data in curves.items():
        ratios = np.asarray(data["ratios"], dtype=float)
        ppl = np.asarray(data["ppl"], dtype=float)
        delta = np.asarray(data.get("ppl_delta_pct", (ppl / ppl[0] - 1) * 100), dtype=float)
        ax1.plot(ratios, ppl, marker="o", markersize=3.2, linewidth=1.4, label=label)
        ax2.plot(ratios, delta, marker="s", markersize=3.0, linewidth=1.3,
                 linestyle="--", label=label)
    ax2.axhline(5.0, color="crimson", linestyle=":", linewidth=1.2)
    ax2.text(0.32, 6.0, "+5% PPL budget", fontsize=7, color="crimson")
    ax1.set_xlabel("Pruning ratio")
    ax1.set_ylabel("WikiText-2 PPL")
    ax1.set_title("(a) Absolute perplexity")
    ax2.set_xlabel("Pruning ratio")
    ax2.set_ylabel("$\\Delta$ PPL (%)")
    ax2.set_title("(b) Relative PPL degradation")
    ax1.legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return save(fig, stem, outdir)
