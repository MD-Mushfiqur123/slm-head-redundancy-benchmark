#!/usr/bin/env python3
"""Phase 5 - empirical calibration.

Computes the Taylor first-order, empirical Fisher, L2 weight-norm and
attention-entropy matrices for one benchmark model and persists per-model
raw telemetry (results/raw/{key}_importance.npz) plus the canonical
results/head_importance_matrix.csv and results/head_entropy_matrix.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data import calibration_and_eval_sequences, load_wikitext_split  # noqa: E402
from src.core.hooks import HeadGatingHookManager  # noqa: E402
from src.core.model_loader import (  # noqa: E402
    MODEL_REGISTRY,
    load_model,
    load_tokenizer,
)
from src.metrics.base_metric import min_max_normalize  # noqa: E402
from src.metrics.entropy import EntropyProfiler  # noqa: E402
from src.metrics.fisher import FisherImportanceScorer  # noqa: E402
from src.metrics.taylor import TaylorImportanceScorer  # noqa: E402
from src.metrics.weight_norm import WeightNormScorer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_batches(sequences, pad_id, batch_size, max_len=None):
    import torch

    entries = [list(s[:max_len]) for s in sequences]
    batches = []
    for i in range(0, len(entries), batch_size):
        batches.append(entries[i : i + batch_size])
    return batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--num-calibration", type=int, default=150)
    args = ap.parse_args()

    spec = MODEL_REGISTRY[args.model]
    out_raw = os.path.join(ROOT, "results", "raw")
    os.makedirs(out_raw, exist_ok=True)

    t_start = time.perf_counter()
    print(f"[calibration] loading {spec.hf_id} ...", flush=True)
    model = load_model(spec)
    tok = load_tokenizer(spec)
    slicing_ok = spec.num_layers == len(
        __import__("src.core.model_loader", fromlist=["locate_layers"]).locate_layers(model)
    )
    assert slicing_ok, "registry/model layer mismatch"

    rows = load_wikitext_split("validation")
    calibration, evaluation = calibration_and_eval_sequences(
        rows, tok, num_calibration=args.num_calibration, num_eval=200,
        window=args.window, min_tokens=50,
    )
    n_cal_tokens = sum(len(s) for s in calibration)
    print(
        f"[calibration] {len(calibration)} calibration seqs ({n_cal_tokens} tokens), "
        f"{len(evaluation)} held-out eval seqs",
        flush=True,
    )

    hm = HeadGatingHookManager(model)
    batches = make_batches(calibration, tok.pad_token_id, args.batch_size)

    taylor = TaylorImportanceScorer()
    fisher = FisherImportanceScorer()
    grads_all = []
    t0 = time.perf_counter()
    for bi, entries in enumerate(batches):
        from src.core.data import autocast_batch

        batch = autocast_batch(entries, tok.pad_token_id)
        hm.reset_gradients()
        out = model(**batch)
        out.loss.backward()
        total_tokens = int((batch["labels"] != -100).sum().item())
        taylor.process_batch(hm, batch)
        fisher.process_batch(hm, batch)
        grads_all.append(hm.get_per_sample_alpha_grads().clone())
        if (bi + 1) % 10 == 0:
            print(
                f"  batch {bi + 1}/{len(batches)} loss={float(out.loss.detach()):.4f} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )

    taylor_matrix = taylor.finalize(n_cal_tokens)  # normalised
    fisher_matrix = fisher.finalize(n_cal_tokens)
    hm.remove_hooks()
    del grads_all

    weight_norm = WeightNormScorer(order=2).compute(model)

    print("[calibration] attention entropy pass ...", flush=True)
    profiler = EntropyProfiler(spec.num_layers, spec.num_heads)
    with torch.no_grad():
        for bi, entries in enumerate(batches):
            from src.core.data import autocast_batch

            batch = autocast_batch(entries, tok.pad_token_id)
            profiler.process_batch(model, batch)
            if (bi + 1) % 25 == 0:
                print(f"  entropy batch {bi + 1}/{len(batches)}", flush=True)
    entropy_matrix = profiler.finalize()

    raw = {
        "taylor_raw": taylor.raw_matrix,
        "taylor_normalized": taylor_matrix,
        "fisher": fisher.raw_matrix,
        "weight_norm_l2": weight_norm,
        "entropy": entropy_matrix,
        "entropy_normalized": min_max_normalize(entropy_matrix),
    }
    npz_path = os.path.join(out_raw, f"{spec.key}_importance.npz")
    np.savez_compressed(npz_path, **raw)

    meta = {
        "model_key": spec.key,
        "hf_id": spec.hf_id,
        "num_layers": spec.num_layers,
        "num_heads": spec.num_heads,
        "d_model": spec.d_model,
        "d_head": spec.d_head,
        "num_calibration_sequences": len(calibration),
        "num_calibration_tokens": n_cal_tokens,
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "calibration_seconds": time.perf_counter() - t_start,
    }
    with open(os.path.join(out_raw, f"{spec.key}_calibration_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[calibration] complete:", npz_path, flush=True)


if __name__ == "__main__":
    main()
