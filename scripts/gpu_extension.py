#!/usr/bin/env python3
"""GPU extension for SLM-HEAD-REDUNDANCY-MONUMENT — standalone CLI version.

Equivalent to colab/slm_head_redundancy_gpu.ipynb but runnable as a plain
Python script (Colab terminal, any CUDA machine, or CPU-only smoke tests):

    python scripts/gpu_extension.py --model gpt2_small --ratio 0.30
    python scripts/gpu_extension.py --model smollm_135m --ratio 0.055556
    python scripts/gpu_extension.py --model all          # full 7-model matrix

Registered architectures: gpt2_small, gpt2_medium, smollm_135m (GQA group 3),
opt_125m, pythia_160m, pythia_410m (fused interleaved QKV), qwen2_05b
(GQA group 7; use --ratio 0.291667). GQA ratios must excise whole KV groups.

Stages (identical mathematics and protocols to the CPU pipeline):
  1. CUDA Taylor + Fisher calibration (150 WikiText-2 windows)
  2. GQA-safe global pruning mask
  3. WikiText-2 / PTB / LAMBADA retention (unpruned vs pruned, gating)
  4. Physical surgery + equivalence check + synchronized GPU wall-clock
     latency B in {1, 4, 8} (Table-6 protocol)
  5. Measured KV-cache bytes from live past_key_values vs Theorem 3

The /TRUTH law applies: every value is measured at runtime; artifacts are
written to results/raw/gpu_*_{model}_{ratio}.json.
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
import transformers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data import (  # noqa: E402
    autocast_batch,
    calibration_and_eval_sequences,
    load_lambada_texts,
    load_ptb_sentences,
    load_wikitext_split,
)
from src.core.custom_dataset import load_custom_rows  # noqa: E402
from src.core.hooks import HeadGatingHookManager  # noqa: E402
from src.core.model_loader import MODEL_REGISTRY, load_model, load_tokenizer  # noqa: E402
from src.eval.downstream import evaluate_lambada  # noqa: E402
from src.eval.perplexity import evaluate_sliding_perplexity  # noqa: E402
from src.metrics.fisher import FisherImportanceScorer  # noqa: E402
from src.metrics.taylor import TaylorImportanceScorer  # noqa: E402
from src.pruners.global_pruner import GlobalHeadPruner  # noqa: E402
from src.pruners.structured_pruner import excise_heads  # noqa: E402
from src.profiler.latency_bench import (  # noqa: E402
    decode_latency_ms_per_token,
    prefill_latency_ms,
)

PAPER_CPU_PPL = {"gpt2_small": 36.6757, "gpt2_medium": 27.0176, "smollm_135m": 24.3911}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model", default="gpt2_small",
        choices=[*list(MODEL_REGISTRY), "all"],
        help="model key, or 'all' to run the full benchmark matrix sequentially",
    )
    ap.add_argument("--ratio", type=float, default=0.30)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--num-calibration", type=int, default=150)
    ap.add_argument("--num-eval", type=int, default=200)
    ap.add_argument("--num-lambada", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--prefill-len", type=int, default=256)
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--lat-repeats", type=int, default=7)
    ap.add_argument("--lat-warmup", type=int, default=2)
    ap.add_argument("--fp16", action="store_true", help="fp16 latency/KV profiling")
    ap.add_argument("--corpus", default="",
                    help="user corpus replacing WikiText-2 for calibration + WT2 eval "
                         "(.txt / .csv / .jsonl / hf:org/name); first windows go to "
                         "calibration, the next ones to evaluation")
    ap.add_argument("--force", action="store_true", help="recompute cached stages")
    ap.add_argument("--no-plots", action="store_true", help="skip matplotlib outputs")
    return ap.parse_args()


def ratio_tag(r: float) -> str:
    return ("%.1f" % (100 * r)).replace(".0", "")


def banner(msg: str) -> None:
    print("─" * 64, flush=True)
    print(f"  {msg}", flush=True)
    print("─" * 64, flush=True)


def free_vram() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def measured_kv_cache_mb(model, ids) -> float:
    out = model(ids, use_cache=True)
    past = out.past_key_values
    if isinstance(past, (tuple, list)):
        tensors = [t for layer in past
                   for t in (layer if isinstance(layer, (tuple, list)) else [layer])
                   if torch.is_tensor(t)]
    elif hasattr(past, "key_cache"):
        tensors = list(past.key_cache) + list(past.value_cache)
    elif hasattr(past, "layers"):
        tensors = []
        for layer in past.layers:
            for attr in ("keys", "values", "key_cache", "value_cache"):
                t = getattr(layer, attr, None)
                if torch.is_tensor(t):
                    tensors.append(t)
    else:
        raise TypeError(type(past))
    return sum(t.numel() * t.element_size() for t in tensors) / 2**20


def main() -> None:
    args = parse_args()
    keys = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    for key in keys:
        args.model = key
        run(args)


def run(args: argparse.Namespace) -> None:
    spec = MODEL_REGISTRY[args.model]
    kv_group = spec.num_heads // spec.num_kv_heads
    if kv_group > 1:
        total_heads = spec.num_layers * spec.num_heads
        excised = int(round(total_heads * args.ratio))
        excised = (excised // kv_group) * kv_group
        if excised == 0 and args.ratio > 0:
            excised = kv_group
        snapped = excised / total_heads
        if snapped != args.ratio:
            print(f"[gqa] {spec.key}: kv_group={kv_group}; snapped ratio "
                  f"{args.ratio:.4f} -> {snapped:.6f} "
                  f"({excised}/{total_heads} heads, group-aligned)")
            args.ratio = snapped
    full_word = args.model != "smollm_135m"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    dtype_bytes = 2 if args.fp16 else 4

    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    def artifact(stage: str) -> str:
        return f"results/raw/gpu_{stage}_{spec.key}_{ratio_tag(args.ratio)}.json"

    def fresh(path: str) -> bool:
        return args.force or not os.path.exists(path)

    torch.manual_seed(7)
    np.random.seed(7)

    banner(f"SLM-HEAD-REDUNDANCY-MONUMENT GPU extension | {spec.hf_id} "
           f"({spec.num_layers}L x {spec.num_heads}H) | ratio {args.ratio:.0%} | "
           f"{'fp16' if args.fp16 else 'fp32'} | {gpu_name}")

    tok = load_tokenizer(spec)
    model = load_model(spec)
    params_millions = sum(p.numel() for p in model.parameters()) / 1e6
    model = model.to(device).eval()

    # ── 1. calibration ────────────────────────────────────────────────────
    imp_npz = f"results/raw/gpu_importance_{spec.key}.npz"
    if not fresh(imp_npz):
        data = np.load(imp_npz)
        taylor_raw, fisher_raw = data["taylor_raw"], data["fisher_raw"]
        calib_seconds = float(data["calib_seconds"]) if "calib_seconds" in data else float("nan")
        banner("cache hit: GPU importance matrices loaded")
    else:
        banner(f"stage 1: CUDA calibration, {args.num_calibration} windows, batch {args.batch_size}")
        if args.corpus:
            rows = load_custom_rows(args.corpus)
            print(f"corpus: {len(rows)} rows from {args.corpus}")
        else:
            rows = load_wikitext_split("validation")
        calibration, _ = calibration_and_eval_sequences(
            rows, tok, num_calibration=args.num_calibration,
            num_eval=args.num_eval, window=args.window, min_tokens=50)
        n_cal_tokens = sum(len(s) for s in calibration)

        rng = np.random.default_rng(7)
        order = rng.permutation(len(calibration))
        batches = [[calibration[j] for j in order[i : i + args.batch_size]]
                   for i in range(0, len(order), args.batch_size)]

        try:
            from tqdm.auto import tqdm
            batches = tqdm(batches, desc="calibration", unit="batch")
        except ImportError:
            pass

        hm = HeadGatingHookManager(model)
        taylor, fisher = TaylorImportanceScorer(), FisherImportanceScorer()
        t0 = time.perf_counter()
        for entries in batches:
            batch = autocast_batch(entries, tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            hm.reset_gradients()
            out = model(**batch)
            out.loss.backward()
            taylor.process_batch(hm, {"labels": batch["labels"].cpu()})
            fisher.process_batch(hm, {"labels": batch["labels"].cpu()})
        calib_seconds = time.perf_counter() - t0
        taylor_raw = taylor.finalize(n_cal_tokens)
        fisher_raw = fisher.finalize(n_cal_tokens)
        hm.reset_gradients()
        hm.remove_hooks()
        free_vram()
        np.savez(imp_npz, taylor_raw=taylor_raw, fisher_raw=fisher_raw,
                 calib_seconds=calib_seconds)
        print(f"calibration: {calib_seconds:.1f}s ({n_cal_tokens/calib_seconds:,.0f} tok/s)")

    # ── 2. mask ───────────────────────────────────────────────────────────
    banner("stage 2: GQA-safe global pruning mask")
    pruner = GlobalHeadPruner(spec.num_layers, spec.num_heads, kv_group_size=kv_group)
    mask = pruner.build_mask(taylor_raw, args.ratio)
    excised = int((mask == 0).sum())
    total_heads = spec.num_layers * spec.num_heads
    active_per_layer = [int((mask[l] == 1).sum()) for l in range(spec.num_layers)]
    print(f"excised {excised}/{total_heads} heads ({100*excised/total_heads:.1f}%), "
          f"kv_group={kv_group}")

    with open(artifact("mask"), "w") as f:
        json.dump({"model": spec.key, "ratio": args.ratio, "kv_group": kv_group,
                   "excised": excised, "total_heads": total_heads,
                   "mask": mask.tolist(), "device": gpu_name}, f, indent=2)

    # ── 3. retention ──────────────────────────────────────────────────────
    ppl_path = artifact("ppl")
    if not fresh(ppl_path):
        results_eval = json.load(open(ppl_path))
        banner("cache hit: retention results")
    else:
        banner("stage 3: WikiText-2 / PTB / LAMBADA on CUDA (gated ablation)")
        if args.corpus:
            wt_rows = load_custom_rows(args.corpus)
        else:
            wt_rows = load_wikitext_split("validation")
        _, evaluation = calibration_and_eval_sequences(
            wt_rows, tok, num_calibration=args.num_calibration,
            num_eval=args.num_eval, window=args.window, min_tokens=50)
        ptb_ids: list = []
        for sent in load_ptb_sentences("test"):
            ptb_ids.extend(tok(sent, add_special_tokens=False)["input_ids"])
        ptb_seq, pos = [], 0
        while pos + 64 < len(ptb_ids) and len(ptb_seq) < 100:
            ptb_seq.append(ptb_ids[pos : pos + args.window])
            pos += args.window
        lambada = load_lambada_texts(args.num_lambada, "test")

        results_eval = {"model": spec.key, "ratio": args.ratio, "device": gpu_name,
                        "corpus": (os.path.basename(args.corpus) if args.corpus
                                   else "wikitext-2-raw-v1"),
                        "protocol": {"window": args.window, "stride": args.stride,
                                     "num_eval": args.num_eval,
                                     "lambada": args.num_lambada}}
        hm = HeadGatingHookManager(model)
        for tag, masked in (("unpruned", False), ("pruned", True)):
            if masked:
                hm.set_mask(mask)
            for name, seqs in (("WT2", evaluation), ("PTB", ptb_seq)):
                r = evaluate_sliding_perplexity(
                    model, seqs, tok.pad_token_id, batch_size=args.batch_size,
                    stride=args.stride, max_length=args.window, device=device)
                results_eval[f"{name.lower()}_{tag}"] = {"ppl": r["perplexity"],
                                                         "tokens": r["total_tokens"]}
                print(f"  {name:6s} {tag:9s} PPL = {r['perplexity']:10.4f}")
            r = evaluate_lambada(model, tok, lambada, device=device, full_word=full_word)
            results_eval[f"lambada_{tag}"] = {k: float(v) for k, v in r.items()}
            print(f"  LAMBADA {tag:7s} top-1 = {100*r['top1_accuracy']:.2f}%")
        hm.clear_mask()
        hm.remove_hooks()
        free_vram()
        json.dump(results_eval, open(ppl_path, "w"), indent=2)

    ref = None if args.corpus else PAPER_CPU_PPL.get(spec.key)
    if ref:
        u = results_eval["wt2_unpruned"]["ppl"]
        print(f"cross-check vs CPU paper baseline: paper={ref}  here={u:.4f}  "
              f"diff={100*(u/ref-1):+.2f}%")

    # ── 4. surgery + latency ──────────────────────────────────────────────
    banner("stage 4: physical surgery + equivalence + synchronized latency")
    pruned_model = excise_heads(model, mask).eval().to(device)

    probe = {"input_ids": torch.randint(0, len(tok), (2, 128), device=device),
             "attention_mask": torch.ones(2, 128, dtype=torch.long, device=device)}
    eq_model, eq_pruned = model, pruned_model
    hm = HeadGatingHookManager(eq_model)
    hm.set_mask(mask)
    with torch.no_grad():
        logits_gated = eq_model(**probe).logits
        logits_surgery = eq_pruned(**probe).logits
    equiv = float((logits_gated - logits_surgery).abs().max())
    equiv_rel = equiv / float(logits_surgery.abs().max())
    hm.clear_mask()
    hm.remove_hooks()
    rel_tol = 2e-3
    assert equiv_rel < rel_tol, f"surgery mismatch: {equiv:.2e} (rel {equiv_rel:.2e})"
    if equiv_rel > 1e-4:
        print(
            f"  note: rel {equiv_rel:.2e} > 1e-4 — expected kernel/reduction-order "
            f"noise when gating and excised matmuls use different cuBLAS shapes"
        )
    print(f"surgery-vs-gating max |Δlogit| = {equiv:.2e} (relative {equiv_rel:.2e}) ✓")
    free_vram()

    lat_path = artifact("latency")
    if not fresh(lat_path):
        lat_results = json.load(open(lat_path))
        banner("cache hit: latency results")
    else:
        prompt = "The study of attention mechanisms in neural networks began with "
        base_ids = tok(prompt, return_tensors="pt")["input_ids"][:, : args.prefill_len].to(device)
        lat_results = {"model": spec.key, "ratio": args.ratio, "fp16": args.fp16,
                       "gpu": gpu_name, "torch": torch.__version__,
                       "surgery_max_logit_diff": equiv,
                       "surgery_rel_logit_diff": equiv_rel,
                       "surgery_rel_tolerance": rel_tol,
                       "protocol": {"decode_steps": args.decode_steps,
                                    "repeats": args.lat_repeats,
                                    "warmup": args.lat_warmup,
                                    "prefill_len": args.prefill_len},
                       "batch_sizes": {}}
        if args.fp16:
            model = model.half().eval()
            pruned_model = pruned_model.half().eval()
        pruned_name = f"pruned_{ratio_tag(args.ratio)}"
        for vname, vmodel in (("unpruned", model), (pruned_name, pruned_model)):
            for b in (1, 4, 8):
                batch_ids = base_ids.expand(b, -1).contiguous()
                dec = decode_latency_ms_per_token(
                    vmodel, batch_ids, decode_steps=args.decode_steps,
                    repeats=args.lat_repeats, warmup=args.lat_warmup)
                pre = prefill_latency_ms(vmodel, batch_ids,
                                         repeats=args.lat_repeats, warmup=args.lat_warmup)
                lat_results["batch_sizes"].setdefault(str(b), {})[vname] = {
                    "decode_ms_per_token": dec["mean_ms_per_token"],
                    "decode_std_ms_per_token": dec["std_ms_per_token"],
                    "prefill_ms": pre["mean_ms"],
                    "prefill_std_ms": pre["std_ms"]}
                print(f"  B={b}  {vname:12s} decode = {dec['mean_ms_per_token']:7.3f} "
                      f"± {dec['std_ms_per_token']:.3f} ms/token")
        for b, d in lat_results["batch_sizes"].items():
            d["speedup_pct"] = 100 * (1 - d[pruned_name]["decode_ms_per_token"]
                                      / d["unpruned"]["decode_ms_per_token"])
            print(f"  B={b}  GPU speedup = {d['speedup_pct']:+.1f}%")
        json.dump(lat_results, open(lat_path, "w"), indent=2)

    # ── 5. measured KV-cache ──────────────────────────────────────────────
    banner("stage 5: measured KV-cache bytes vs Theorem 3")
    ctx_limit = getattr(model.config, "n_positions", None) or \
        getattr(model.config, "max_position_embeddings", 1024)
    seq_grid = [t for t in (128, 256, 512, 768, 1024, 2048) if t <= ctx_limit]

    texts = [r["text"] for r in load_wikitext_split("validation") if len(r["text"]) > 200]
    probe_text = " ".join(texts)
    long_ids = tok(probe_text, return_tensors="pt")["input_ids"]
    while long_ids.shape[1] < max(seq_grid):
        probe_text = probe_text + " " + probe_text
        long_ids = tok(probe_text, return_tensors="pt")["input_ids"]

    def theorem3_mb(heads_per_layer: list, T: int) -> float:
        return sum(2 * 1 * T * n_l * spec.d_head * dtype_bytes
                   for n_l in heads_per_layer) / 2**20

    kv_path = artifact("kv")
    kv_results = {"model": spec.key, "gpu": gpu_name, "dtype_bytes": dtype_bytes,
                  "fp16": args.fp16, "seq_lens": {}}
    mem_rows = []
    for T in seq_grid:
        ids_T = long_ids[:, :T].contiguous().to(device)
        row = {"T": int(T)}
        for tag, m in (("unpruned", model), ("pruned30", pruned_model)):
            row[tag] = measured_kv_cache_mb(m, ids_T)
        row["reduction_pct"] = 100 * (1 - row["pruned30"] / row["unpruned"])
        row["theoretical_MB"] = theorem3_mb(active_per_layer, T)
        row["unpruned_theoretical_MB"] = theorem3_mb([spec.num_heads] * spec.num_layers, T)
        kv_results["seq_lens"][str(T)] = row
        mem_rows.append(row)
        print(f"  T={T:5d}  unpruned {row['unpruned']:8.2f} MB   "
              f"pruned {row['pruned30']:8.2f} MB   measured −{row['reduction_pct']:.1f}%   "
              f"analytic {row['theoretical_MB']:8.2f} MB")
    json.dump(kv_results, open(kv_path, "w"), indent=2)

    # ── 6. summary ────────────────────────────────────────────────────────
    summary_path = artifact("summary")
    summary = {
        "spec": "SLM-HEAD-REDUNDANCY-MONUMENT — GPU extension",
        "model": spec.key, "hf_id": spec.hf_id,
        "params_millions": round(params_millions, 2), "gpu": gpu_name,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "ratio": args.ratio, "excised_heads": excised, "total_heads": total_heads,
        "calibration_seconds": calib_seconds,
        "retention": results_eval, "latency": lat_results, "kv_cache": kv_results,
        "equivalence": {"surgery_vs_gating_max_logit_diff": equiv,
                        "relative": equiv_rel, "threshold": rel_tol,
                        "note": "relative logit max-diff between gated and "
                                "physically-excised model; evaluated in fp32 "
                                "regardless of --fp16 (fp16 is scoped to "
                                "latency/KV profiling only; tolerance 2e-3)"},
        "truth_law": "every value measured at runtime; no mocks",
    }
    json.dump(summary, open(summary_path, "w"), indent=2)
    banner(f"summary written -> {summary_path}")

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        bs = sorted(int(b) for b in lat_results["batch_sizes"])
        pruned_name = f"pruned_{ratio_tag(args.ratio)}"
        dec_un = [lat_results["batch_sizes"][str(b)]["unpruned"]["decode_ms_per_token"] for b in bs]
        dec_pr = [lat_results["batch_sizes"][str(b)][pruned_name]["decode_ms_per_token"] for b in bs]
        xpos, width = np.arange(len(bs)), 0.36
        ax.bar(xpos - width / 2, dec_un, width, label="unpruned", color="#4472C4")
        ax.bar(xpos + width / 2, dec_pr, width, label=f"pruned @ {args.ratio:.0%}", color="#ED7D31")
        for xi, b in enumerate(bs):
            sp = lat_results["batch_sizes"][str(b)]["speedup_pct"]
            ax.text(xi, max(dec_un[xi], dec_pr[xi]) * 1.02, f"{sp:+.1f}%",
                    ha="center", fontsize=9)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"B={b}" for b in bs])
        ax.set_ylabel("decode latency (ms/token)")
        ax.set_title(f"Wall-clock decode latency — {spec.hf_id} on {gpu_name}")
        ax.legend()
        fig.savefig(f"results/figures/gpu_latency_{spec.key}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        ts = [r["T"] for r in mem_rows]
        ax.plot(ts, [r["unpruned"] for r in mem_rows], "o-", label="measured unpruned",
                color="#C00000")
        ax.plot(ts, [r["pruned30"] for r in mem_rows], "s-", label=f"measured pruned @{args.ratio:.0%}",
                color="#2E7D32")
        ax.plot(ts, [r["theoretical_MB"] for r in mem_rows], "--", label="Theorem 3 (pruned)",
                color="#7CB342", alpha=0.8)
        ax.set_xlabel("sequence length T")
        ax.set_ylabel("KV-cache footprint (MB)")
        ax.set_title(f"Live KV-cache bytes vs Theorem 3 — {spec.hf_id}")
        ax.legend()
        fig.savefig(f"results/figures/gpu_kv_{spec.key}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("figures: results/figures/gpu_{latency,kv}_%s.png" % spec.key)

    free_vram()


if __name__ == "__main__":
    main()
