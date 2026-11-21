#!/usr/bin/env python3
"""Phase 6 - progressive pruning sweep for one model / one importance metric.

For every ratio P in the sweep the script:
  1. builds the binary head mask from an importance matrix,
  2. evaluates WikiText-2 (200 held-out sequences) and Penn Treebank (100
     sequences) sliding-window perplexity in float32 CPU,
  3. performs physical weight surgery and measures decode latency (B=1) plus
     analytic KV-cache footprint (Theorem 3),
  4. optionally evaluate LAMBADA zero-shot completion.

Raw telemetry goes to results/raw/sweep_{model}_{metric}.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data import (  # noqa: E402
    calibration_and_eval_sequences,
    load_lambada_texts,
    load_ptb_sentences,
    load_wikitext_split,
)
from src.core.hooks import HeadGatingHookManager  # noqa: E402
from src.core.model_loader import (  # noqa: E402
    coupon_shuffle_mask,
    load_model,
    load_tokenizer,
    MODEL_REGISTRY,
)
from src.eval.downstream import evaluate_lambada  # noqa: E402
from src.eval.perplexity import evaluate_sliding_perplexity  # noqa: E402
from src.pruners.base_pruner import expected_pruned_count  # noqa: E402
from src.pruners.global_pruner import GlobalHeadPruner  # noqa: E402
from src.pruners.structured_pruner import excise_heads  # noqa: E402
from src.profiler.latency_bench import decode_latency_ms_per_token  # noqa: E402
from src.profiler.memory_tracker import kv_cache_footprint_mb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RATIOS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]


def load_importance(spec, metric: str, raw_dir: str) -> np.ndarray:
    path = os.path.join(raw_dir, f"{spec.key}_importance.npz")
    data = np.load(path)
    if metric == "taylor":
        return data["taylor_raw"]
    if metric == "fisher":
        return data["fisher"]
    if metric == "weight_norm_l2":
        return data["weight_norm_l2"]
    raise ValueError(metric)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--metric", default="taylor",
                    choices=["taylor", "fisher", "weight_norm_l2", "random"])
    ap.add_argument("--ratios", type=float, nargs="*", default=DEFAULT_RATIOS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[13, 29, 47, 61, 83],
                    help="seeds for the random control baseline")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--num-calibration", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lambada", action="store_true",
                    help="also evaluate zero-shot LAMBADA at every ratio")
    ap.add_argument("--num-lambada", type=int, default=250)
    ap.add_argument("--no-full-word", action="store_true",
                    help="skip the greedy full-word check (faster LAMBADA)")
    ap.add_argument("--latency-repeats", type=int, default=5)
    ap.add_argument("--latency-steps", type=int, default=24)
    ap.add_argument("--suffix", default=None,
                    help="override the output filename suffix (default: metric)")
    args = ap.parse_args()

    spec = MODEL_REGISTRY[args.model]
    raw_dir = os.path.join(ROOT, "results", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    kv_group = 3 if spec.num_kv_heads != spec.num_heads else 1

    model = load_model(spec)
    tok = load_tokenizer(spec)

    wt_rows = load_wikitext_split("validation")
    calibration, eval_seqs = calibration_and_eval_sequences(
        wt_rows, tok, num_calibration=args.num_calibration, num_eval=200,
        window=args.window, min_tokens=50,
    )
    n_ptb = 100
    ptb_sentences = load_ptb_sentences("test")
    ptb_ids: list = []
    for sent in ptb_sentences:
        ptb_ids.extend(tok(sent, add_special_tokens=False)["input_ids"])
    # 100 consecutive ~window-token sequences over the PTB corpus
    ptb_seqs, pos = [], 0
    while pos + 64 < len(ptb_ids) and len(ptb_seqs) < n_ptb:
        ptb_seqs.append(ptb_ids[pos : pos + args.window])
        pos += args.window
    texts = list(eval_seqs) + list(ptb_seqs)

    pruner = GlobalHeadPruner(spec.num_layers, spec.num_heads, kv_group_size=kv_group)
    importance = None if args.metric == "random" else load_importance(spec, args.metric, raw_dir)
    rng = np.random.default_rng(13)

    lambada_texts = load_lambada_texts(args.num_lambada) if args.lambada else None

    telemetry = []
    for ratio in args.ratios:
        ratio = round(ratio, 6)
        # ---------- random control: fresh mask per seed, then mean ----------
        if args.metric == "random":
            ppl_runs = []
            for seed in args.seeds:
                mask = coupon_shuffle_mask(
                    spec.num_layers, spec.num_heads, ratio,
                    np.random.default_rng(seed), kv_group_size=kv_group,
                )
                hm = HeadGatingHookManager(model)
                hm.set_mask(mask)
                res = evaluate_sliding_perplexity(
                    model, texts, tok.pad_token_id, batch_size=args.batch_size,
                    stride=args.stride, max_length=args.window, device="cpu",
                )
                hm.clear_mask()
                hm.remove_hooks()
                ppl_runs.append(res["perplexity"])
            telemetry.append({
                "pruning_ratio": ratio,
                "metric": "random",
                "wikitext_ppl": float(np.mean(ppl_runs)),
                "wikitext_ppl_std": float(np.std(ppl_runs, ddof=1)) if len(ppl_runs) > 1 else 0.0,
                "ptb_ppl": float("nan"),
                "active_heads": int(round(spec.num_layers * spec.num_heads - ratio * spec.num_layers * spec.num_heads)),
                "pruned_heads": expected_pruned_count(spec.num_layers, spec.num_heads, ratio),
            })
            gc.collect()
            continue

        mask = pruner.build_mask(importance, ratio)
        assert (mask == 0).sum() == expected_pruned_count(spec.num_layers, spec.num_heads, ratio)

        # ---- masked evaluation (exact ablation, no weight surgery) ----
        hm = HeadGatingHookManager(model)
        hm.set_mask(mask)
        res_wt = evaluate_sliding_perplexity(
            model, eval_seqs, tok.pad_token_id, batch_size=args.batch_size,
            stride=args.stride, max_length=args.window, device="cpu",
        )
        res_ptb = evaluate_sliding_perplexity(
            model, ptb_seqs, tok.pad_token_id, batch_size=args.batch_size,
            stride=args.stride, max_length=args.window, device="cpu",
        )
        lambada_res = None
        if lambada_texts is not None:
            lambada_res = evaluate_lambada(
                model, tok, lambada_texts, device="cpu",
                full_word=not args.no_full_word,
            )
        hm.clear_mask()
        hm.remove_hooks()

        # ---- physical surgery -> latency + KV telemetry ----
        pruned_model = excise_heads(model, mask)
        prompt_ids = tok("The meaning of life is", return_tensors="pt")["input_ids"]
        lat = decode_latency_ms_per_token(
            pruned_model, prompt_ids, decode_steps=args.latency_steps,
            repeats=args.latency_repeats, warmup=2,
        )
        active_by_layer = [int(mask[li].sum()) for li in range(spec.num_layers)]
        kv_unique = []
        base_kv = spec.num_kv_heads if spec.num_kv_heads != spec.num_heads else spec.num_heads
        gs = kv_group
        for li in range(spec.num_layers):
            kept_groups = sum(
                1 for g0 in range(0, spec.num_heads, gs) if mask[li, g0 : g0 + gs].any()
            )
            kv_unique.append(kept_groups)
        kv_unpruned = kv_cache_footprint_mb(
            spec.num_layers, spec.num_kv_heads, spec.d_head, batch_size=8,
            seq_len=1024, dtype_bytes=2,
        )
        kv_pruned = kv_cache_footprint_mb(
            spec.num_layers, int(np.mean(kv_unique)), spec.d_head, batch_size=8,
            seq_len=1024, dtype_bytes=2,
        )
        del pruned_model
        gc.collect()

        row = {
            "pruning_ratio": ratio,
            "metric": args.metric,
            "active_heads": int(mask.sum()),
            "pruned_heads": int((mask == 0).sum()),
            "wikitext_ppl": res_wt["perplexity"],
            "wikitext_total_tokens": res_wt["total_tokens"],
            "wikitext_eval_seconds": res_wt["eval_time_seconds"],
            "ptb_ppl": res_ptb["perplexity"],
            "latency_ms_per_token": lat["mean_ms_per_token"],
            "latency_std_ms_per_token": lat["std_ms_per_token"],
            "kv_cache_mb_B8_T1024": kv_pruned,
            "kv_cache_unpruned_mb_B8_T1024": kv_unpruned,
            "kv_active_groups_per_layer": kv_unique,
        }
        if lambada_res is not None:
            row["lambada_top1"] = lambada_res["top1_accuracy"]
            row["lambada_full_word_acc"] = lambada_res["full_word_accuracy"]
            row["lambada_target_word_ppl"] = lambada_res["target_word_ppl"]
        telemetry.append(row)
        print(f"[sweep {spec.key}/{args.metric}] P={ratio:.2f} "
              f"wt_ppl={row['wikitext_ppl']:.4f} ptb={row['ptb_ppl']:.4f} "
              f"lat={row['latency_ms_per_token']:.3f}ms", flush=True)
        if lambada_res is not None:
            print(f"    lambada top1={100*row['lambada_top1']:.2f}% "
                  f"full_word={100*row['lambada_full_word_acc']:.2f}%", flush=True)

    out = {
        "model_key": spec.key,
        "hf_id": spec.hf_id,
        "metric": args.metric,
        "protocol": {
            "wikitext_sequences": 200, "ptb_sequences": 100,
            "stride": args.stride, "max_length": args.window,
            "num_lambada": args.num_lambada if args.lambada else 0,
            "dtype": "fp32-cpu", "kv_scenario": "B=8, T=1024, fp16 bytes",
        },
        "rows": telemetry,
    }
    suffix = args.suffix if args.suffix else args.metric
    path = os.path.join(raw_dir, f"sweep_{spec.key}_{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("[sweep] saved", path, flush=True)


if __name__ == "__main__":
    main()
