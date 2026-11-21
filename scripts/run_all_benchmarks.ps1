# Master automation wrapper (Windows PowerShell): full benchmark suite.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"

python -m unittest discover tests/
python run_experiments.py
python scripts/aggregate_results.py
python scripts/generate_figures.py
Write-Host "Benchmark suite complete. Telemetry in results/, figures in results/figures/."
