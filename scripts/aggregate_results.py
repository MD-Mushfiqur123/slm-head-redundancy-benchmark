#!/usr/bin/env python3
"""Aggregate raw telemetry into the canonical result CSVs.

Outputs:
  results/head_importance_matrix.csv   (GPT-2 Small canonical + all models)
  results/head_entropy_matrix.csv
  results/pruning_curve.csv
  results/downstream_results.csv
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.model_loader import MODEL_REGISTRY  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
OUT = os.path.join(ROOT, "results")


def build_importance_csv() -> None:
    frames = []
    for key in MODEL_REGISTRY:
        path = os.path.join(RAW, f"{key}_importance.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        spec = MODEL_REGISTRY[key]
        rows = []
        for li in range(spec.num_layers):
            for h in range(spec.num_heads):
                rows.append({
                    "model": key,
                    "layer": li,
                    "head": h,
                    "taylor_raw": float(data["taylor_raw"][li, h]),
                    "taylor_normalized": float(data["taylor_normalized"][li, h]),
                    "fisher": float(data["fisher"][li, h]),
                    "l2_weight_norm": float(data["weight_norm_l2"][li, h]),
                    "l1_weight_norm": float(data.get("weight_norm_l1", data["weight_norm_l2"])[li, h]),
                    "entropy_bits": float(data["entropy"][li, h]),
                    "entropy_normalized": float(data["entropy_normalized"][li, h]),
                })
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(OUT, "head_importance_matrix.csv"), index=False)
    # canonical single-model matrices (GPT-2 Small)
    canonical = df[df.model == "gpt2_small"].drop(columns=["model"])
    canonical[["layer", "head", "taylor_normalized", "fisher", "l2_weight_norm"]].to_csv(
        os.path.join(OUT, "head_importance_matrix_gpt2_small.csv"), index=False
    )
    canonical[["layer", "head", "entropy_bits", "entropy_normalized"]].to_csv(
        os.path.join(OUT, "head_entropy_matrix.csv"), index=False
    )


def build_pruning_curve_csv() -> None:
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "sweep_*.json"))):
        with open(path) as f:
            blob = json.load(f)
        for row in blob["rows"]:
            frames.append({"model": blob["model_key"], "metric": blob["metric"], **{
                k: v for k, v in row.items()
            }})
    if not frames:
        return
    df = pd.DataFrame(frames)
    df.to_csv(os.path.join(OUT, "pruning_curve.csv"), index=False)


def build_downstream_csv() -> None:
    frames = []
    # LAMBADA telemetry embedded in the Taylor sweeps
    for path in sorted(glob.glob(os.path.join(RAW, "sweep_*.json"))):
        with open(path) as f:
            blob = json.load(f)
        for row in blob["rows"]:
            if "lambada_top1" in row:
                frames.append({
                    "model": blob["model_key"],
                    "metric": blob["metric"],
                    "pruning_ratio": row["pruning_ratio"],
                    "lambada_top1": row["lambada_top1"],
                    "lambada_full_word_acc": row.get("lambada_full_word_acc"),
                    "target_word_ppl": row.get("lambada_target_word_ppl"),
                    "num_examples": blob["protocol"]["num_lambada"],
                })
    if not frames:
        return
    df = pd.DataFrame(frames)
    df.to_csv(os.path.join(OUT, "downstream_results.csv"), index=False)


def build_ablation_csv() -> None:
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "ablation_*.json"))):
        with open(path) as f:
            blob = json.load(f)
        for method, res in blob["results"].items():
            frames.append({
                "model": blob["model_key"],
                "pruning_ratio": blob["ratio"],
                "method": method,
                "wikitext_ppl_mean": res["wikitext_ppl_mean"],
                "wikitext_ppl_std": res["wikitext_ppl_std"],
                "lambada_top1_mean": res["lambada_top1_mean"],
                "lambada_top1_std": res["lambada_top1_std"],
                "latency_mean_ms": res["latency_mean"],
                "latency_std_ms": res["latency_std"],
                "num_seeds": len(res["per_seed"]),
            })
    if not frames:
        return
    pd.DataFrame(frames).to_csv(os.path.join(OUT, "ablation_results.csv"), index=False)


def build_layer_stratification() -> None:
    """Stratified pruning statistics for Table 4 (GPT-2 Small)."""
    path = os.path.join(RAW, "gpt2_small_importance.npz")
    if not os.path.exists(path):
        return
    from src.pruners.global_pruner import GlobalHeadPruner

    data = np.load(path)
    imp = data["taylor_raw"]
    p = GlobalHeadPruner(12, 12)
    m30 = p.build_mask(imp, 0.30)
    m50 = p.build_mask(imp, 0.50)
    strata = [("Early (L0--L3)", 0, 4), ("Intermediate (L4--L7)", 4, 8),
              ("Deep (L8--L11)", 8, 12)]
    out = {"total_heads": 144}
    for name, lo, hi in strata:
        out[name] = {
            "total_heads": (hi - lo) * 12,
            "excised_30": int((m30[lo:hi] == 0).sum()),
            "excised_50": int((m50[lo:hi] == 0).sum()),
        }
    out["excised_30_total"] = int((m30 == 0).sum())
    out["excised_50_total"] = int((m50 == 0).sum())
    with open(os.path.join(OUT, "layer_stratification.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    build_importance_csv()
    build_pruning_curve_csv()
    build_downstream_csv()
    build_ablation_csv()
    build_layer_stratification()
    print("[aggregate] CSVs written to", OUT)
    for name in ("head_importance_matrix.csv", "head_entropy_matrix.csv",
                 "pruning_curve.csv", "downstream_results.csv",
                 "ablation_results.csv"):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            print("  -", name, len(pd.read_csv(p)), "rows")
