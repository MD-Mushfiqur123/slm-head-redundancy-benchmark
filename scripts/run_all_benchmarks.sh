#!/usr/bin/env bash
# Master automation wrapper: full benchmark suite from a clean checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=0

python -m unittest discover tests/
python run_experiments.py
python scripts/aggregate_results.py
python scripts/generate_figures.py
echo "Benchmark suite complete. Telemetry in results/, figures in results/figures/."
