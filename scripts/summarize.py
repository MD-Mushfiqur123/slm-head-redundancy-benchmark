#!/usr/bin/env python3
"""Produce results/summary.json with the headline statistics for the paper."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def main():
    out = {}
    curve = pd.read_csv(os.path.join(RES, "pruning_curve.csv"))

    g = curve[(curve.model == "gpt2_small") & (curve.metric == "taylor")].sort_values("pruning_ratio")
    base = float(g[g.pruning_ratio == 0.0].iloc[0].wikitext_ppl)
    out["gpt2_small_base_ppl"] = base
    out["gpt2_small_ptb_base_ppl"] = float(g[g.pruning_ratio == 0.0].iloc[0].ptb_ppl)
    for r in sorted(g.pruning_ratio.unique()):
        row = g[g.pruning_ratio == r].iloc[0]
        rr = round(float(r), 2)
        out[f"gpt2_small_p{int(rr*100)}"] = {
            "wikitext_ppl": float(row.wikitext_ppl),
            "ppl_delta_pct": 100 * (float(row.wikitext_ppl) - base) / base,
            "ptb_ppl": float(row.ptb_ppl),
            "latency_ms": float(row.latency_ms_per_token),
            "speedup_pct": None,
            "lambada_top1_pct": (100 * float(row.lambada_top1)
                                 if "lambada_top1" in row and row["lambada_top1"] == row["lambada_top1"] else None),
            "lambada_full_word_pct": (100 * float(row.lambada_full_word_acc)
                                      if "lambada_full_word_acc" in row and row["lambada_full_word_acc"] == row["lambada_full_word_acc"] else None),
            "kv_cache_mb": float(row.kv_cache_mb_B8_T1024),
            "kv_reduction_pct": 100 * (float(row.kv_cache_unpruned_mb_B8_T1024) - float(row.kv_cache_mb_B8_T1024))
                                / float(row.kv_cache_unpruned_mb_B8_T1024),
        }
    base_lat = out["gpt2_small_p0"]["latency_ms"]
    for k, v in out.items():
        if isinstance(v, dict) and "latency_ms" in v:
            v["speedup_pct"] = 100 * (base_lat - v["latency_ms"]) / base_lat
    # max retainable sparsity under 5% budget
    for model in ("gpt2_small", "gpt2_medium", "smollm_135m"):
        gg = curve[(curve.model == model) & (curve.metric == "taylor")].sort_values("pruning_ratio")
        if gg.empty:
            continue
        b = float(gg[gg.pruning_ratio == 0.0].iloc[0].wikitext_ppl)
        budget = 0.05 * b
        maxr = max([float(r) for r in gg.pruning_ratio
                    if float(gg[gg.pruning_ratio == r].iloc[0].wikitext_ppl) - b <= budget], default=0.0)
        out[f"{model}_max_ratio_under_5pct"] = maxr
        out[f"{model}_base_ppl"] = b
        for r in (0.2, 0.4):
            row = gg[gg.pruning_ratio == r]
            if not row.empty:
                out[f"{model}_p{int(r*100)}"] = float(row.iloc[0].wikitext_ppl)

    strat_path = os.path.join(RES, "layer_stratification.json")
    if os.path.exists(strat_path):
        with open(strat_path) as f:
            out["stratification"] = json.load(f)

    imp_path = os.path.join(RES, "raw", "gpt2_small_importance.npz")
    if os.path.exists(imp_path):
        data = np.load(imp_path)
        out["gpt2_small_layer_mean_taylor"] = {int(li): float(data["taylor_raw"][li].mean() * 1e4)
                                               for li in range(12)}
        shares = data["taylor_raw"].sum(1) / data["taylor_raw"].sum() * 100
        out["gpt2_small_stratum_share_pct"] = {
            "early_L0_3": round(float(shares[:4].sum()), 2),
            "intermediate_L4_7": round(float(shares[4:8].sum()), 2),
            "deep_L8_11": round(float(shares[8:].sum()), 2),
        }
        ent_corr = np.corrcoef(data["entropy"].ravel(), data["taylor_normalized"].ravel())[0, 1]
        out["entropy_taylor_pearson"] = float(ent_corr)

    for key in ("gpt2_medium", "smollm_135m"):
        p = os.path.join(RES, "raw", f"{key}_importance.npz")
        if os.path.exists(p):
            data = np.load(p)
            imp = data["taylor_raw"]
            shares = imp.sum(1) / imp.sum() * 100
            nl = imp.shape[0]
            third = nl // 4
            out[f"{key}_stratum_share_pct"] = {
                "shallow_first_quarter": round(float(shares[:third].sum()), 2),
                "intermediate_middle_half": round(float(shares[third:-third].sum()), 2),
                "deep_last_quarter": round(float(shares[-third:].sum()), 2),
            }

    ab = os.path.join(RES, "ablation_results.csv")
    if os.path.exists(ab):
        df = pd.read_csv(ab)
        df = df[df.model == "gpt2_small"]
        t = df[df.method == "taylor"].iloc[0]
        out["ablation_30"] = {
            m: {
                "wikitext_ppl_mean": float(df[df.method == m].iloc[0].wikitext_ppl_mean),
                "wikitext_ppl_std": float(df[df.method == m].iloc[0].wikitext_ppl_std),
                "lambada_top1_mean_pct": 100 * float(df[df.method == m].iloc[0].lambada_top1_mean),
                "lambada_top1_std_pct": 100 * float(df[df.method == m].iloc[0].lambada_top1_std),
                "latency_mean_ms": float(df[df.method == m].iloc[0].latency_mean_ms),
                "latency_std_ms": float(df[df.method == m].iloc[0].latency_std_ms),
            } for m in df.method.unique()
        }

    lat_path = os.path.join(RES, "raw", "latency_gpt2_small.json")
    if os.path.exists(lat_path):
        with open(lat_path) as f:
            lat = json.load(f)
        out["latency_table6"] = lat["batch_sizes"]
        out["kv_cache_curves"] = lat["kv_cache_curves"]

    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
