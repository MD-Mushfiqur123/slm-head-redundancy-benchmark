#!/usr/bin/env python3
"""Weekly regression guard: compare quick-protocol PPL drift."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-drift", type=float, default=0.5)
    args = ap.parse_args()
    sweeps = sorted(glob.glob(os.path.join(ROOT, "results", "raw", "sweep_gpt2_small_taylor.json")))
    if not sweeps:
        print("[regression] no telemetry found; run the sweep first")
        sys.exit(1)
    with open(sweeps[-1]) as f:
        rows = json.load(f)["rows"]
    by_ratio = {round(r["pruning_ratio"], 2): r["wikitext_ppl"] for r in rows}
    base = by_ratio.get(0.0)
    pruned = by_ratio.get(0.3)
    if base is None or pruned is None:
        print("[regression] missing baseline or 30% row")
        sys.exit(1)
    drift = abs(pruned - base)
    print(f"[regression] baseline PPL={base:.4f}, 30% PPL={pruned:.4f}, drift={drift:.4f}")
    sys.exit(0 if drift <= args.max_drift else 2)


if __name__ == "__main__":
    main()
