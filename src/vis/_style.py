"""Publication-grade figures: shared style helpers."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig, stem: str, outdir: str = "results/figures"):
    """Save vector PDF and 300-DPI PNG versions of a figure."""
    import os

    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, stem + ".pdf")
    png_path = os.path.join(outdir, stem + ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path
