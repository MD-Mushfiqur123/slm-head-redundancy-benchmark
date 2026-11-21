#!/usr/bin/env python3
"""Phase 6c - hardware latency and throughput benchmark (Table 6 / Fig. 5).

Measures wall-clock decode latency (ms/token) across batch sizes B in
{1, 4, 8} for the unpruned GPT-2 Small and its 30% Taylor-pruned variant
(physical weight surgery), plus empirical KV-cache RSS deltas.
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

from src.core.model_loader import MODEL_REGISTRY, load_model, load_tokenizer  # noqa: E402
from src.pruners.global_pruner import GlobalHeadPruner  # noqa: E402
from src.pruners.structured_pruner import excise_heads  # noqa: E402
from src.profiler.latency_bench import decode_latency_ms_per_token, prefill_latency_ms  # noqa: E402
from src.profiler.memory_tracker import (  # noqa: E402
    empirical_kv_memory_mb,
    kv_cache_footprint_mb,
    peak_rss_mb,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2_small", choices=list(MODEL_REGISTRY))
    ap.add_argument("--ratio", type=float, default=0.30)
    ap.add_argument("--prefill-len", type=int, default=256)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--kv-seq-lens", type=int, nargs="*",
                    default=[128, 256, 512, 768, 1024, 2048])
    args = ap.parse_args()

    spec = MODEL_REGISTRY[args.model]
    raw_dir = os.path.join(ROOT, "results", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    kv_group = 3 if spec.num_kv_heads != spec.num_heads else 1

    tok = load_tokenizer(spec)
    prompt = "The study of attention mechanisms in neural networks began with "
    ids = tok(prompt, return_tensors="pt")["input_ids"][:, : args.prefill_len]

    results = {"model_key": spec.key, "prefill_len": args.prefill_len,
               "decode_steps": args.decode_steps, "batch_sizes": {}}

    variants = {}
    model = load_model(spec)
    variants["unpruned"] = (model, None, spec.num_kv_heads)

    data = np.load(os.path.join(raw_dir, f"{spec.key}_importance.npz"))
    pruner = GlobalHeadPruner(spec.num_layers, spec.num_heads, kv_group_size=kv_group)
    mask = pruner.build_mask(data["taylor_raw"], args.ratio)
    variants[f"pruned_{int(args.ratio*100)}"] = (excise_heads(model, mask), mask, None)

    for name, (mdl, maskarr, nkv_default) in variants.items():
        per_batch = {}
        for b in (1, 4, 8):
            batch_ids = ids.expand(b, -1).contiguous()
            dec = decode_latency_ms_per_token(
                mdl, batch_ids, decode_steps=args.decode_steps,
                repeats=args.repeats, warmup=args.warmup,
            )
            pre = prefill_latency_ms(mdl, batch_ids, repeats=args.repeats, warmup=args.warmup)
            per_batch[b] = {
                "decode_ms_per_token": dec["mean_ms_per_token"],
                "decode_std": dec["std_ms_per_token"],
                "prefill_ms": pre["mean_ms"],
                "prefill_std": pre["std_ms"],
                "decode_steps": args.decode_steps,
            }
            print(f"[latency {name} B={b}] decode={per_batch[b]['decode_ms_per_token']:.3f} "
                  f"ms/tok prefill={per_batch[b]['prefill_ms']:.1f} ms", flush=True)
        results["batch_sizes"][name] = per_batch

    # effective KV heads per layer (unique, after surgery)
    def nkv_effective(maskarr):
        if maskarr is None:
            return [spec.num_kv_heads] * spec.num_layers
        group = spec.num_heads // spec.num_kv_heads
        return [sum(1 for g0 in range(0, spec.num_heads, group) if maskarr[li, g0:g0+group].any())
                for li in range(spec.num_layers)]

    # analytical KV memory vs sequence length (B=8, fp16) + empirical RSS
    kv_curve = {"seq_lengths": args.kv_seq_lens}
    context_limit = int(getattr(spec, "max_position_embeddings", 0) or
                        getattr(model.config, "max_position_embeddings", 1024))
    for name, (mdl, maskarr, _) in variants.items():
        pruned_heads = None
        if maskarr is not None:
            pruned_heads = {li: [int(h) for h in np.where(maskarr[li] == 0)[0]]
                            for li in range(spec.num_layers)
                            if (maskarr[li] == 0).sum() > 0}
        an = []
        emp = []
        for T in args.kv_seq_lens:
            nkv = int(np.mean(nkv_effective(maskarr)))
            an.append(kv_cache_footprint_mb(spec.num_layers, nkv, spec.d_head,
                                            batch_size=8, seq_len=T, dtype_bytes=2))
            if T <= context_limit:
                emp.append(empirical_kv_memory_mb(mdl, tok, prompt, seq_len=T,
                                                  pruned_heads=pruned_heads))
            else:
                emp.append(None)
        kv_curve[name] = {"analytical_mb": an, "empirical_peak_rss_mb": emp}
        print(f"[kv {name}] analytic={an[:3]}... empirical={emp[:3]}...", flush=True)
    results["kv_cache_curves"] = kv_curve

    results["peak_rss_mb_total"] = peak_rss_mb()
    path = os.path.join(raw_dir, f"latency_{spec.key}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("[latency] saved", path, flush=True)


if __name__ == "__main__":
    main()
