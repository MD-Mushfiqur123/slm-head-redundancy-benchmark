#!/usr/bin/env python3
"""End-to-end reproduction of every experiment behind the manuscript.

Runs, in order:
  Phase 5: gradient/entropy calibration for all three models
  Phase 6: progressive Taylor sweeps (+ LAMBADA) for all three models,
           Fisher / L2-norm / random control sweeps for GPT-2 Small,
           fine-grained sub-10% Taylor grids for all three models
  Phase 6b: 4-way paradigm ablation at 30% (GPT-2 Small)
  Phase 6c: hardware latency / KV-cache benchmarks
  Phase 7-8: aggregation, figures, tables, and table embedding into
           paper/main.tex

Full execution takes several hours on a single x86-64 CPU core
(all acceptance runs used a 4.2 GB memory limit).
"""
from __future__ import annotations

import subprocess
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd, desc):
    print(f"\n>>> {desc}: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


PHASES = [
    (["python", "scripts/run_calibration.py", "--model", "gpt2_small", "--batch-size", "1"],
     "Phase 5a: calibration / importance matrices, GPT-2 Small"),
    (["python", "scripts/run_calibration.py", "--model", "smollm_135m", "--batch-size", "1"],
     "Phase 5b: calibration, SmolLM-135M"),
    (["python", "scripts/run_calibration.py", "--model", "gpt2_medium", "--batch-size", "1", "--window", "384"],
     "Phase 5c: calibration, GPT-2 Medium"),
    (["python", "scripts/run_sweep.py", "--model", "gpt2_small", "--metric", "taylor",
      "--lambada", "--latency-repeats", "7"],
     "Phase 6a: progressive Taylor sweep + LAMBADA, GPT-2 Small"),
    (["python", "scripts/run_sweep.py", "--model", "smollm_135m", "--metric", "taylor",
      "--lambada", "--no-full-word", "--latency-repeats", "5"],
     "Phase 6b: progressive Taylor sweep + LAMBADA, SmolLM-135M"),
    (["python", "scripts/run_sweep.py", "--model", "gpt2_medium", "--metric", "taylor",
      "--lambada", "--no-full-word", "--latency-repeats", "5"],
     "Phase 6c: progressive Taylor sweep + LAMBADA, GPT-2 Medium"),
    (["python", "scripts/run_sweep.py", "--model", "gpt2_small", "--metric", "fisher",
      "--latency-repeats", "3"],
     "Phase 6d: Fisher sweep, GPT-2 Small"),
    (["python", "scripts/run_sweep.py", "--model", "gpt2_small", "--metric",
      "weight_norm_l2", "--latency-repeats", "3"],
     "Phase 6e: L2 weight-norm sweep, GPT-2 Small"),
    (["python", "scripts/run_sweep.py", "--model", "gpt2_small", "--metric", "random",
      "--latency-repeats", "3"],
     "Phase 6f: random control sweep (5 seeds), GPT-2 Small"),
    (["python", "scripts/run_sweep.py", "--model", "gpt2_small", "--metric", "taylor",
      "--ratios", "0.02", "0.05", "0.08", "--suffix", "taylor_fine",
      "--latency-repeats", "3"],
     "Phase 6g: fine-grained Taylor grid, GPT-2 Small"),
    (["python", "scripts/run_sweep.py", "--model", "smollm_135m", "--metric", "taylor",
      "--ratios", "0.0222222", "0.0555555", "0.0777777", "--suffix", "taylor_fine",
      "--latency-repeats", "3", "--no-full-word"],
     "Phase 6h: fine-grained Taylor grid (KV-group-aligned), SmolLM-135M"),
    (["python", "scripts/run_sweep.py", "--model", "gpt2_medium", "--metric", "taylor",
      "--ratios", "0.02", "0.05", "0.08", "--suffix", "taylor_fine",
      "--latency-repeats", "3"],
     "Phase 6i: fine-grained Taylor grid, GPT-2 Medium"),
    (["python", "scripts/run_ablation.py", "--model", "gpt2_small", "--batch-size", "4"],
     "Phase 6j: 4-way paradigm ablation at 30%"),
    (["python", "scripts/run_latency.py", "--model", "gpt2_small"],
     "Phase 6k: hardware latency & KV-cache profiling"),
    (["python", "scripts/summarize.py"],
     "Phase 7a: summary statistics"),
    (["python", "scripts/aggregate_results.py"],
     "Phase 7b: CSV aggregation"),
    (["python", "scripts/generate_figures.py"],
     "Phase 7c: vector figure rendering"),
    (["python", "scripts/make_tables.py"],
     "Phase 8a: table generation"),
    (["python", "scripts/embed_tables.py"],
     "Phase 8b: embed measured table bodies into paper/main.tex"),
]


def main():
    for cmd, desc in PHASES:
        run(cmd, desc)
    print("\nALL EXPERIMENT PHASES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
