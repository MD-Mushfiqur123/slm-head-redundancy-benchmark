"""Figures 1 and 4: head importance heatmaps and entropy-sensitivity scatter."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ._style import apply_style, save


def plot_importance_heatmap(
    importance: np.ndarray,
    title: str,
    stem: str,
    outdir: str = "results/figures",
    xlabel: str = "Head index",
    ylabel: str = "Layer index",
):
    """Figure 1: [L x Nh] viridis heatmap of normalised head importance."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    im = ax.imshow(importance, cmap="viridis", aspect="auto")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Normalised Taylor importance")
    fig.tight_layout()
    return save(fig, stem, outdir)


def plot_entropy_scatter(
    entropy: np.ndarray,
    taylor: np.ndarray,
    stem: str,
    outdir: str = "results/figures",
):
    """Figure 4: attention entropy vs Taylor sensitivity scatter with
    Pearson correlation and least-squares trend line."""
    import matplotlib.pyplot as plt
    from scipy import stats

    apply_style()
    x = entropy.reshape(-1)
    y = taylor.reshape(-1)
    r, p_value = stats.pearsonr(x, y)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.scatter(x, y, s=18, c="#2a6fb0", alpha=0.8, edgecolors="k", linewidths=0.3)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, "r--", linewidth=1.2,
            label=f"fit: slope={slope:.2f}")
    ax.set_xlabel("Mean attention entropy (bits)")
    ax.set_ylabel("Normalised Taylor importance")
    ax.set_title(f"Entropy vs. sensitivity (Pearson r = {r:.3f}, p = {p_value:.1e})")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return save(fig, stem, outdir), r, p_value
