#!/usr/bin/env python3
"""Phase 7 - render all six publication figures (PDF vector + 300-DPI PNG)."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.model_loader import MODEL_REGISTRY  # noqa: E402
from src.vis.cross_model import plot_retention_curves  # noqa: E402
from src.vis.entropy_scatter import plot_kv_memory  # noqa: E402
from src.vis.heatmaps import plot_entropy_scatter, plot_importance_heatmap  # noqa: E402
from src.vis.pareto import plot_pareto  # noqa: E402
from src.vis.vulnerability import plot_layer_survival, stratify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
FIGS = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGS, exist_ok=True)


def load_sweep(model_key: str, metric: str):
    path = os.path.join(RAW, f"sweep_{model_key}_{metric}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def figure1():
    data = np.load(os.path.join(RAW, "gpt2_small_importance.npz"))
    imp = data["taylor_normalized"]
    plot_importance_heatmap(
        imp,
        title="Attention head sensitivity, GPT-2 Small (124M)",
        stem="figure1_importance_heatmap",
        outdir=FIGS,
    )
    # per-model heatmaps as bonus raw outputs
    for key in ("gpt2_medium", "smollm_135m"):
        p = os.path.join(RAW, f"{key}_importance.npz")
        if os.path.exists(p):
            plot_importance_heatmap(
                np.load(p)["taylor_normalized"],
                title=f"Head sensitivity, {MODEL_REGISTRY[key].hf_id}",
                stem=f"figure1_importance_heatmap_{key}",
                outdir=FIGS,
            )


def figure2():
    sweeps = {}
    for metric, label in (("taylor", "Taylor (proposed)"), ("fisher", "Fisher"),
                          ("weight_norm_l2", "L2 weight-norm"), ("random", "Random (mean ± 1σ)")):
        blob = load_sweep("gpt2_small", metric)
        if blob is None:
            continue
        rows = sorted(blob["rows"], key=lambda r: r["pruning_ratio"])
        curves = {"ratios": [r["pruning_ratio"] for r in rows],
                  "ppl": [r["wikitext_ppl"] for r in rows]}
        if metric == "random":
            curves["ppl_std"] = [r.get("wikitext_ppl_std", 0.0) for r in rows]
        sweeps[label] = curves
    if sweeps:
        plot_pareto(
            sweeps, stem="figure2_pareto_frontier", outdir=FIGS,
            title="WikiText-2 PPL vs. pruning ratio (GPT-2 Small, 124M)",
            annotate_threshold=0.30,
        )


def figure3():
    path = os.path.join(RAW, "gpt2_small_importance.npz")
    if not os.path.exists(path):
        return
    imp = np.load(path)["taylor_raw"]
    p = GlobalHeadPrunerLike(12, 12)
    masks = {}
    for ratio, label in ((0.30, "30% pruned"), (0.50, "50% pruned")):
        mask = p.build_mask(imp, ratio)
        masks[label] = mask.sum(axis=1) / 12.0
    plot_layer_survival(
        masks,
        strata=[("Early (L0-L3)", 4), ("Intermediate (L4-L7)", 8), ("Deep (L8-L11)", 12)],
        stem="figure3_layer_vulnerability",
        outdir=FIGS,
    )


def GlobalHeadPrunerLike(nl, nh):
    from src.pruners.global_pruner import GlobalHeadPruner

    return GlobalHeadPruner(nl, nh)


def figure4():
    data = np.load(os.path.join(RAW, "gpt2_small_importance.npz"))
    paths, r, p = plot_entropy_scatter(
        data["entropy"], data["taylor_normalized"],
        stem="figure4_entropy_scatter", outdir=FIGS,
    )
    with open(os.path.join(ROOT, "results", "entropy_correlation.json"), "w") as f:
        json.dump({"pearson_r": float(r), "p_value": float(p)}, f, indent=2)


def figure5():
    path = os.path.join(RAW, "latency_gpt2_small.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        blob = json.load(f)
    curve = blob["kv_cache_curves"]
    xs = curve["seq_lengths"]
    plot_kv_memory(
        xs,
        curve["unpruned"]["analytical_mb"],
        {k: curve[k]["analytical_mb"] for k in curve if k.startswith("pruned")},
        stem="figure5_kv_cache_memory",
        outdir=FIGS,
    )
    # empirical RSS companion figure
    import matplotlib.pyplot as plt
    from src.vis._style import apply_style, save

    apply_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    def _plot_emp(xs, ys, **kw):
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            return
        gx, gy = zip(*pts)
        ax.plot(gx, gy, **kw)
    _plot_emp(xs, curve["unpruned"]["empirical_peak_rss_mb"],
              marker="o", label="Unpruned (measured)")
    for k in curve:
        if k.startswith("pruned"):
            _plot_emp(xs, curve[k]["empirical_peak_rss_mb"], marker="s", linestyle="--",
                      label=f"{k} (measured)")
    ax.set_xlabel("Sequence length T (tokens)")
    ax.set_ylabel("Peak RSS delta (MB)")
    ax.set_title("Empirical memory scaling (B=1, fp32)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "figure5b_kv_empirical_rss", FIGS)


def figure6():
    curves = {}
    for key in ("gpt2_small", "gpt2_medium", "smollm_135m"):
        blob = load_sweep(key, "taylor")
        if blob is None:
            continue
        rows = sorted(blob["rows"], key=lambda r: r["pruning_ratio"])
        label = {"gpt2_small": "GPT-2 Small (124M)", "gpt2_medium": "GPT-2 Medium (355M)",
                 "smollm_135m": "SmolLM-135M"}[key]
        ppl = np.array([r["wikitext_ppl"] for r in rows])
        curves[label] = {
            "ratios": [r["pruning_ratio"] for r in rows],
            "ppl": ppl.tolist(),
            "ppl_delta_pct": (100 * (ppl / ppl[0] - 1)).tolist(),
        }
    if curves:
        plot_retention_curves(curves, stem="figure6_cross_model_retention", outdir=FIGS)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    print("[figures] written to", FIGS)
    for f in sorted(glob.glob(os.path.join(FIGS, "*"))):
        print("  -", os.path.basename(f))
