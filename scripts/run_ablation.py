#!/usr/bin/env python3
"""Phase 6b - four-way pruning paradigm ablation at the 30% threshold.

Methods: Taylor (proposed), Fisher, L2 weight-norm, Random (5 seeds).
Every method is evaluated on WikiText-2 perplexity, LAMBADA zero-shot
completion and decode latency (physical surgery, B=1).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data import (  # noqa: E402
    calibration_and_eval_sequences,
    load_lambada_texts,
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
from src.pruners.global_pruner import GlobalHeadPruner  # noqa: E402
from src.pruners.structured_pruner import excise_heads  # noqa: E402
from src.profiler.latency_bench import decode_latency_ms_per_token  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2_small", choices=list(MODEL_REGISTRY))
    ap.add_argument("--ratio", type=float, default=0.30)
    ap.add_argument("--seeds", type=int, nargs="*", default=[13, 29, 47, 61, 83])
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--num-calibration", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-lambada", type=int, default=250)
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
    lambada_texts = load_lambada_texts(args.num_lambada)
    pruner = GlobalHeadPruner(spec.num_layers, spec.num_heads, kv_group_size=kv_group)
    data = np.load(os.path.join(raw_dir, f"{spec.key}_importance.npz"))

    methods = {
        "taylor": ("taylor_raw", None),
        "fisher": ("fisher", None),
        "weight_norm_l2": ("weight_norm_l2", None),
        "random": (None, args.seeds),
    }
    results = {}
    for method, (matrix_key, seeds) in methods.items():
        if matrix_key is not None:
            mask = pruner.build_mask(data[matrix_key], args.ratio)
            seed_results = [mask]
        else:
            seed_results = [
                coupon_shuffle_mask(
                    spec.num_layers, spec.num_heads, args.ratio,
                    np.random.default_rng(seed), kv_group_size=kv_group,
                )
                for seed in seeds
            ]
        per_seed = []
        for si, mask in enumerate(seed_results):
            hm = HeadGatingHookManager(model)
            hm.set_mask(mask)
            res_wt = evaluate_sliding_perplexity(
                model, eval_seqs, tok.pad_token_id, batch_size=args.batch_size,
                stride=args.stride, max_length=args.window, device="cpu",
            )
            res_lam = evaluate_lambada(model, tok, lambada_texts, device="cpu")
            hm.clear_mask()
            hm.remove_hooks()
            pruned_model = excise_heads(model, mask)
            ids = tok("The meaning of life is", return_tensors="pt")["input_ids"]
            lat = decode_latency_ms_per_token(pruned_model, ids, decode_steps=24, repeats=5, warmup=2)
            del pruned_model
            gc.collect()
            per_seed.append({
                "seed": seeds[si] if seeds else 0,
                "wikitext_ppl": res_wt["perplexity"],
                "lambada_top1": res_lam["top1_accuracy"],
                "lambada_full_word_acc": res_lam["full_word_accuracy"],
                "latency_ms_per_token": lat["mean_ms_per_token"],
            })
            print(f"[ablation {method} seed {si}] ppl={per_seed[-1]['wikitext_ppl']:.4f} "
                  f"lambada={100*per_seed[-1]['lambada_top1']:.2f}% "
                  f"lat={per_seed[-1]['latency_ms_per_token']:.3f}ms", flush=True)
        ppls = [r["wikitext_ppl"] for r in per_seed]
        lams = [r["lambada_top1"] for r in per_seed]
        lats = [r["latency_ms_per_token"] for r in per_seed]
        results[method] = {
            "per_seed": per_seed,
            "wikitext_ppl_mean": float(np.mean(ppls)),
            "wikitext_ppl_std": float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0,
            "lambada_top1_mean": float(np.mean(lams)),
            "lambada_top1_std": float(np.std(lams, ddof=1)) if len(lams) > 1 else 0.0,
            "latency_mean": float(np.mean(lats)),
            "latency_std": float(np.std(lats, ddof=1)) if len(lats) > 1 else 0.0,
        }
    # baseline (unpruned) reference row
    out = {
        "model_key": spec.key,
        "ratio": args.ratio,
        "num_lambada": args.num_lambada,
        "random_seeds": args.seeds,
        "results": results,
    }
    path = os.path.join(raw_dir, f"ablation_{spec.key}_p{int(args.ratio*100)}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("[ablation] saved", path, flush=True)


if __name__ == "__main__":
    main()
