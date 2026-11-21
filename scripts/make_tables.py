#!/usr/bin/env python3
"""Phase 8 helper - transcribe measured telemetry into LaTeX tables.

Every value in paper/tables/table*.tex is generated directly from
results/*.csv (or the raw sweep latency telemetry), guaranteeing the
4-decimal congruence demanded by the audit protocol.
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
RES = os.path.join(ROOT, "results")
TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(TAB, exist_ok=True)


def f4(x):
    return f"{float(x):.4f}"


def f2(x):
    return f"{float(x):.2f}"


def f1(x):
    return f"{float(x):.1f}"


def f3(x):
    return f"{float(x):.3f}"


def write(name, content):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(content)
    print("wrote paper/tables/" + name)


def table1():
    df = pd.read_csv(os.path.join(RES, "pruning_curve.csv"))
    g = df[(df.model == "gpt2_small") & (df.metric == "taylor")].sort_values("pruning_ratio")
    base = g[g.pruning_ratio == 0.0].iloc[0]
    base_ppl = float(base.wikitext_ppl)
    base_lat = float(base.latency_ms_per_token)
    kv_unpruned = float(base.kv_cache_unpruned_mb_B8_T1024)
    rows = []
    for _, r in g.iterrows():
        ppl = float(r.wikitext_ppl)
        lat = float(r.latency_ms_per_token)
        kv = float(r.kv_cache_mb_B8_T1024) if r.pruning_ratio > 0 else kv_unpruned
        rows.append(
            f"{100*float(r.pruning_ratio):.0f}\\% & {int(r.active_heads)} & "
            f"{int(r.pruned_heads)} & {f4(ppl)} & {f2(100*(ppl-base_ppl)/base_ppl)} & "
            f"{f3(lat)} & {f1(100*(base_lat-lat)/base_lat)} & {f2(kv)} \\\\"
        )
    body = "\n".join(rows)
    write("table1.tex", body)


def table2():
    df = pd.read_csv(os.path.join(RES, "ablation_results.csv"))
    df = df[(df.model == "gpt2_small")]
    order = {"taylor": "Taylor first-order (proposed)",
             "fisher": "Empirical Fisher",
             "weight_norm_l2": "L2 weight-norm",
             "random": "Random (control)"}
    rows = []
    for key, label in order.items():
        r = df[df.method == key]
        if r.empty:
            continue
        r = r.iloc[0]
        if key == "random":
            ppl_s = f"{f4(r.wikitext_ppl_mean)} $\\pm$ {f4(r.wikitext_ppl_std)}"
            lam_s = f"{f2(100*r.lambada_top1_mean)} $\\pm$ {f2(100*r.lambada_top1_std)}"
            lat_s = f"{f3(r.latency_mean_ms)} $\\pm$ {f3(r.latency_std_ms)}"
        else:
            ppl_s = f4(r.wikitext_ppl_mean)
            lam_s = f2(100 * r.lambada_top1_mean)
            lat_s = f3(r.latency_mean_ms)
        rows.append(f"{label} & {ppl_s} & {lam_s} & {lat_s} \\\\")
    write("table2.tex", "\n".join(rows))


def table3():
    df = pd.read_csv(os.path.join(RES, "pruning_curve.csv"))
    labels = {"gpt2_small": "GPT-2 Small (124M)",
              "gpt2_medium": "GPT-2 Medium (355M)",
              "smollm_135m": "SmolLM-135M (135M)"}
    rows = []
    for key in ("gpt2_small", "gpt2_medium", "smollm_135m"):
        g = df[(df.model == key) & (df.metric == "taylor")].sort_values("pruning_ratio")
        if g.empty:
            continue
        base = float(g[g.pruning_ratio == 0.0].iloc[0].wikitext_ppl)
        p20 = float(g[g.pruning_ratio == 0.2].iloc[0].wikitext_ppl)
        p40 = float(g[g.pruning_ratio == 0.4].iloc[0].wikitext_ppl)
        budget = 0.05 * base
        max_ratio = 0.0
        for _, r in g.iterrows():
            if float(r.wikitext_ppl) - base <= budget:
                max_ratio = max(max_ratio, float(r.pruning_ratio))
        rows.append(
            f"{labels[key]} & {f4(base)} & {f4(p20)} & {f4(p40)} & "
            f"{100*max_ratio:.0f}\\% \\\\"
        )
    write("table3.tex", "\n".join(rows))


def table4():
    path = os.path.join(RES, "layer_stratification.json")
    with open(path) as f:
        blob = json.load(f)
    rows = []
    for stratum in ("Early (L0--L3)", "Intermediate (L4--L7)", "Deep (L8--L11)"):
        d = blob[stratum]
        rows.append(
            f"{stratum} & {d['total_heads']} & {d['excised_30']} & "
            f"{d['excised_50']} & {f1(100.0 - d['excised_30'] / d['total_heads'] * 100)} \\\\"
        )
    rows.append(
        f"\\textbf{{All (L0--L11)}} & {blob['total_heads']} & "
        f"{blob['excised_30_total']} & {blob['excised_50_total']} & "
        f"{f1(100.0 - 100.0 * blob['excised_30_total'] / blob['total_heads'])} \\\\"
    )
    write("table4.tex", "\n".join(rows))


def table5():
    df = pd.read_csv(os.path.join(RES, "downstream_results.csv"))
    g = df[(df.model == "gpt2_small") & (df.metric == "taylor")].sort_values("pruning_ratio")
    rows = []
    for _, r in g.iterrows():
        rows.append(
            f"{100*float(r.pruning_ratio):.0f}\\% & {f2(100*float(r.lambada_top1))} & "
            f"{f2(100*float(r.lambada_full_word_acc))} & {f3(r.target_word_ppl)} \\\\"
        )
    write("table5.tex", "\n".join(rows))


def table6():
    path = os.path.join(RES, "raw", "latency_gpt2_small.json")
    with open(path) as f:
        blob = json.load(f)
    bs = blob["batch_sizes"]
    rows = []
    for b in (1, 4, 8):
        u = bs["unpruned"][str(b)] if str(b) in bs["unpruned"] else bs["unpruned"][b]
        p = bs["pruned_30"][str(b)] if str(b) in bs["pruned_30"] else bs["pruned_30"][b]
        speed = 100 * (u["decode_ms_per_token"] - p["decode_ms_per_token"]) / u["decode_ms_per_token"]
        rows.append(
            f"{b} & {f3(u['decode_ms_per_token'])} $\\pm$ {f3(u['decode_std'])} & "
            f"{f3(p['decode_ms_per_token'])} $\\pm$ {f3(p['decode_std'])} & "
            f"{f1(speed)} \\\\"
        )
    write("table6.tex", "\n".join(rows))


if __name__ == "__main__":
    table1()
    table2()
    table3()
    table4()
    table5()
    table6()
