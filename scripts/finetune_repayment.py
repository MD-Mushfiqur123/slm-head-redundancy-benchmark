"""Fine-tuning repayment experiment: how much of the excision damage does a
small fine-tuning budget repair?

Protocol (measured only, no extrapolation):
  1. calibrate Taylor importance on the calibration corpus (num_calibration
     windows of the evaluation corpus split, exactly like the GPU extension);
  2. for each ratio in --ratios: build the GQA-safe global mask, apply physical
     surgery on a fresh model copy and measure the sliding-window perplexity
     of the surgically pruned model (pre-repayment);
  3. fine-tune each surgically pruned model (and optionally an unpruned
     control) on the training corpus with AdamW and cosine-constant schedule
     and measure the post-repayment perplexity;
  4. write results/raw/repayment_<model>.json and
     results/repayment_<model>.csv.

Usage (GPU recommended):
  python scripts/finetune_repayment.py --model gpt2_small \
      --ratios 0.1 0.2 0.3 0.4 --epochs 2 --corpus data/my_corpus.txt
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.data import (  # noqa: E402
    autocast_batch,
    calibration_and_eval_sequences,
    load_wikitext_split,
)
from src.core.custom_dataset import load_custom_rows  # noqa: E402
from src.core.hooks import HeadGatingHookManager  # noqa: E402
from src.core.model_loader import MODEL_REGISTRY, load_model, load_tokenizer  # noqa: E402
from src.eval.perplexity import evaluate_sliding_perplexity  # noqa: E402
from src.metrics.taylor import TaylorImportanceScorer  # noqa: E402
from src.pruners.structured_pruner import excise_heads  # noqa: E402
from src.pruners.global_pruner import GlobalHeadPruner  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="fine-tuning repayment study")
    ap.add_argument("--model", default="gpt2_small", choices=list(MODEL_REGISTRY))
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-windows", type=int, default=2000)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--num-calibration", type=int, default=150)
    ap.add_argument("--num-eval", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7,
                    help="torch/numpy seed; varies calibration batch order and FT")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for output JSON/CSV filenames, e.g. _e4")
    ap.add_argument("--corpus", default="",
                    help="optional user corpus (.txt/.csv/.jsonl/hf:...) "
                         "replacing WikiText-2 for calibration + evaluation")
    ap.add_argument("--train-corpus", default="",
                    help="corpus for fine-tuning; default: --corpus when given, "
                         "otherwise the WikiText-2 train split (disjoint from eval)")
    ap.add_argument("--ft-unpruned", action="store_true",
                    help="also fine-tune an unpruned control model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args()


def banner(msg: str) -> None:
    print("─" * 64, flush=True)
    print(f"  {msg}", flush=True)
    print("─" * 64, flush=True)


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


@torch.no_grad()
def eval_ppl(model, evaluation, pad_id, args, device) -> dict:
    r = evaluate_sliding_perplexity(
        model, evaluation, pad_id, batch_size=args.batch_size,
        stride=args.stride, max_length=args.window, device=device)
    return {"ppl": r["perplexity"], "tokens": r["total_tokens"]}


def snap_ratio(spec, ratio: float) -> float:
    kv_group = spec.num_heads // spec.num_kv_heads
    if kv_group <= 1:
        return ratio
    total = spec.num_layers * spec.num_heads
    excised = (int(round(total * ratio)) // kv_group) * kv_group
    if excised == 0 and ratio > 0:
        excised = kv_group
    return excised / total


def fine_tune(model, train_seqs, pad_id, args, device, desc: str) -> float:
    """AdamW fine-tune, returns wall-clock seconds."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(7)
    order = rng.permutation(len(train_seqs))
    batches = [[train_seqs[j] for j in order[i:i + args.batch_size]]
               for i in range(0, len(order), args.batch_size)]
    total_steps = max(len(batches) * args.epochs, 1)
    warmup = max(int(args.warmup_frac * total_steps), 1)
    step, t0 = 0, time.perf_counter()
    for _epoch in range(args.epochs):
        for batch_ids in batches:
            lr = cosine_lr(step, total_steps, warmup, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = autocast_batch(batch_ids, pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"  [{desc}] step {step}/{total_steps} "
                      f"loss={float(out.loss.detach()):.4f} lr={lr:.2e}", flush=True)
    model.eval()
    return time.perf_counter() - t0


def main() -> None:
    args = parse_args()
    spec = MODEL_REGISTRY[args.model]
    kv_group = spec.num_heads // spec.num_kv_heads
    device = torch.device(args.device)

    os.makedirs("results/raw", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_json = f"results/raw/repayment_{spec.key}{args.out_suffix}.json"
    banner(f"fine-tuning repayment | {spec.hf_id} | ratios {args.ratios} | "
           f"{device.type} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    tok = load_tokenizer(spec)
    if args.corpus:
        rows = load_custom_rows(args.corpus)
        print(f"corpus: {len(rows)} rows from {args.corpus}")
    else:
        rows = load_wikitext_split("validation")
    calibration, evaluation = calibration_and_eval_sequences(
        rows, tok, num_calibration=args.num_calibration, num_eval=args.num_eval,
        window=args.window, min_tokens=50)
    train_source = args.train_corpus or args.corpus
    if train_source:
        train_rows = load_custom_rows(train_source)
    else:
        train_rows = load_wikitext_split("train")
    train_stream = []
    for row in train_rows:
        train_stream.extend(tok(row["text"], add_special_tokens=False)["input_ids"])
    train_seqs = [train_stream[p:p + args.window]
                  for p in range(0, len(train_stream) - args.window + 1,
                                 args.window)][:args.train_windows]
    if not train_seqs:
        raise ValueError(
            f"fine-tuning corpus produced 0 windows of {args.window} tokens "
            f"({len(train_stream)} tokens available); supply a larger corpus")
    print(f"calibration {len(calibration)} | eval {len(evaluation)} | "
          f"train {len(train_seqs)} windows "
          f"({len(train_seqs) * args.window:,} tokens) from "
          f"{train_source or 'wikitext-2 train split'}")

    base = load_model(spec).to(device).eval()
    wt_eval = evaluation
    pad_id = tok.pad_token_id

    banner("stage 0: baseline calibration (Taylor importance)")
    hm = HeadGatingHookManager(base)
    taylor = TaylorImportanceScorer()
    n_cal_tokens = sum(len(s) for s in calibration)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(calibration))
    cal_batches = [[calibration[j] for j in order[i:i + args.batch_size]]
                   for i in range(0, len(order), args.batch_size)]
    t0 = time.perf_counter()
    for entries in cal_batches:
        batch = autocast_batch(entries, pad_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        hm.reset_gradients()
        out = base(**batch)
        out.loss.backward()
        taylor.process_batch(hm, {"labels": batch["labels"].cpu()})
    taylor_raw = taylor.finalize(n_cal_tokens)
    hm.reset_gradients()
    hm.remove_hooks()
    calib_seconds = time.perf_counter() - t0
    gc.collect()
    print(f"calibration: {calib_seconds:.1f}s")

    pruner = GlobalHeadPruner(spec.num_layers, spec.num_heads, kv_group_size=kv_group)
    unpruned_ppl = eval_ppl(base, wt_eval, pad_id, args, device)
    print(f"unpruned PPL = {unpruned_ppl['ppl']:.4f}")

    results = {
        "model": spec.key, "hf_id": spec.hf_id, "device": device.type,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "corpus": (os.path.basename(args.corpus) if args.corpus else "wikitext-2-raw-v1"),
        "protocol": {"epochs": args.epochs, "train_windows": len(train_seqs),
                     "batch_size": args.batch_size, "lr": args.lr,
                     "window": args.window, "num_eval": args.num_eval,
                     "train_corpus": train_source or "wikitext-2 train split"},
        "unpruned_ppl": unpruned_ppl["ppl"],
        "seed": args.seed,
        "out_suffix": args.out_suffix,
        "rows": [],
    }
    if args.ft_unpruned:
        banner("control: fine-tuning the unpruned model")
        ft_seconds = fine_tune(base, train_seqs, pad_id, args, device, "unpruned")
        ctl = eval_ppl(base, wt_eval, pad_id, args, device)
        results["unpruned_finetuned"] = {"ppl": ctl["ppl"], "seconds": ft_seconds}
        print(f"unpruned + FT PPL = {ctl['ppl']:.4f}")
        base = base.cpu()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for ratio in args.ratios:
        ratio = snap_ratio(spec, ratio)
        rate = f"{100 * ratio:.0f}%"
        banner(f"repayment @ excision {rate}")
        pruned = excise_heads(copy.deepcopy(base), pruner.build_mask(taylor_raw, ratio))
        pruned.to(device).eval()
        pre = eval_ppl(pruned, wt_eval, pad_id, args, device)
        print(f"pre-repayment PPL = {pre['ppl']:.4f}")
        ft_seconds = fine_tune(pruned, train_seqs, pad_id, args, device, rate)
        post = eval_ppl(pruned, wt_eval, pad_id, args, device)
        if pre["ppl"] > unpruned_ppl["ppl"]:
            repair = 100 * (1 - (post["ppl"] - unpruned_ppl["ppl"])
                            / (pre["ppl"] - unpruned_ppl["ppl"]))
            print(f"post-repayment PPL = {post['ppl']:.4f} "
                  f"(repair {repair:.1f}% of the damage)")
        else:
            repair = 0.0
            print(f"post-repayment PPL = {post['ppl']:.4f}")
        results["rows"].append({
            "ratio": ratio, "pre_ppl": pre["ppl"], "post_ppl": post["ppl"],
            "pre_delta_pct": 100 * (pre["ppl"] / unpruned_ppl["ppl"] - 1),
            "post_delta_pct": 100 * (post["ppl"] / unpruned_ppl["ppl"] - 1),
            "repair_pct": repair,
            "ft_seconds": ft_seconds,
        })
        json.dump(results, open(out_json, "w"), indent=2)
        pruned = pruned.cpu()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json.dump(results, open(out_json, "w"), indent=2)
    import csv as _csv

    with open(f"results/repayment_{spec.key}{args.out_suffix}.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["ratio", "pre_ppl", "post_ppl", "pre_delta_pct",
                    "post_delta_pct", "repair_pct", "ft_seconds"])
        for r in results["rows"]:
            w.writerow([r["ratio"], r["pre_ppl"], r["post_ppl"], r["pre_delta_pct"],
                        r["post_delta_pct"], r["repair_pct"], r["ft_seconds"]])
    print("artifacts:", out_json, f"+ results/repayment_{spec.key}{args.out_suffix}.csv")
    banner("done — /TRUTH: every value measured at runtime")


if __name__ == "__main__":
    main()
