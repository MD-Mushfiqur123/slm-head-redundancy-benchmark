# SLM-HEAD-REDUNDANCY-MONUMENT

**Layer-Wise Attention Head Redundancy and Sensitivity Geometry in Sub-Billion Language Models: An Empirical Pruning and Hardware Profiling Benchmark**

Author: **Md. Mushfiqur Rahim** — Department of Science, Govt. Azizul Haque College, Bogura, Bangladesh.

This repository is the complete, reproducible implementation of an empirical study of
attention-head redundancy in sub-billion-parameter language models (GPT-2 Small 124M,
GPT-2 Medium 355M, SmolLM-135M). It contains:

- exact-gradient **first-order Taylor** head-sensitivity scoring,
- **empirical Fisher** (second-order) head scoring,
- calibration-free **L2 weight-norm** and **random** control baselines,
- **zero-shot structural head excision** (no fine-tuning, no calibration data at inference),
- verification that physical weight surgery is *bit-equivalent* to gated ablation,
- sliding-window perplexity (WikiText-2 + Penn Treebank), zero-shot LAMBADA completion,
- wall-clock decode-latency benchmarks and Theorem-3 KV-cache memory accounting,
- publication figures and an 8-page IEEE manuscript (`paper/main.tex`).

## The /TRUTH law

Every number in `results/` and every table in the paper is produced by executed code
(`results/raw/*.json` telemetry → `results/*.csv` → `paper/tables/*.tex`). Nothing is
hand-written or extrapolated. Re-run `python run_experiments.py` to regenerate all
telemetry end-to-end.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m unittest discover tests/          # unit tests (22 tests)
python run_experiments.py                   # full benchmark suite (hours)
python scripts/aggregate_results.py         # CSVs
python scripts/generate_figures.py          # Figures 1-6 (PDF + PNG)
python scripts/make_tables.py               # LaTeX tables from CSVs
```

Reduced-protocol smoke run (minutes):

```bash
python scripts/run_calibration.py --model gpt2_small --num-calibration 24
python scripts/run_sweep.py --model gpt2_small --metric taylor --ratios 0.0 0.3 \
  --lambada --num-lambada 24 --num-calibration 24 --latency-repeats 2
```

Interactive exploration:

```bash
streamlit run demo/app.py
```

## GPU extension on Google Colab

The CPU-only sandbox produced no GPU wall-clock telemetry. `colab/slm_head_redundancy_gpu.ipynb`
addresses this on a free Colab T4 runtime:

- CUDA Taylor/Fisher calibration (150 WikiText-2 windows, full backward passes),
- synchronized `torch.cuda.synchronize()` wall-clock decode latency at $B\\in\\{1,4,8\\}$,
  unpruned vs. 30% physically excised model,
- KV-cache footprint measured **directly from the live `past_key_values` tensors**
  (cross-validating Theorem 3) across the context-length grid,
- GQA-safe global mask, surgery-vs-gating equivalence check, WikiText-2 / PTB / LAMBADA retention.

Open directly in Colab:
`https://colab.research.google.com/github/MD-Mushfiqur123/slm-head-redundancy-benchmark/blob/main/colab/slm_head_redundancy_gpu.ipynb`

Prefer the terminal / a plain CUDA machine? The same pipeline ships as a standalone CLI:

```bash
python scripts/gpu_extension.py --model gpt2_small --ratio 0.30
python scripts/gpu_extension.py --model smollm_135m --ratio 0.055556
```

The registry now covers **seven sub-billion architectures**, spanning four parameter
families: GPT-2 Small/Medium (Post-LN, absolute), **SmolLM-135M** (LLaMA family,
RoPE, SwiGLU, GQA group 3), **OPT-125M** (pre-LN, learned positions),
**Pythia-160m/410m** (GPT-NeoX, RoPE, fused interleaved QKV with dedicated
physical surgery), and **Qwen2.5-0.5B** (RoPE, SwiGLU, GQA group 7). GQA models
take a group-aligned ratio (e.g. `--ratio 0.291667` for Qwen2.5-0.5B; 0.30 for
SmolLM). Run the whole matrix sequentially with `--model all` (resumable via the
per-stage artifacts, so an interrupted run loses only the current stage).

All stages carry per-model/per-ratio caches in `results/raw/` (`--force` recomputes).

Everything the notebook prints is measured on the runtime (/TRUTH law); results land in
`results/raw/gpu_*.json` and can be pushed back to a `gpu-telemetry` branch with the
optional final cell (user's own fine-grained PAT, never stored).

## Method summary

1. **Gating formalisation.** Every head $(l,h)$ receives a binary gate
   $\alpha_{l,h}$; multi-head self-attention becomes
   $\mathrm{MHA}_l(X;\boldsymbol\alpha_l)=\sum_h \alpha_{l,h}\,
   \mathrm{Head}_{l,h}(X)W^O_{l,h}$.
2. **Sensitivity geometry.** A forward/backward pass over calibration batches
   yields the exact directional derivative
   $g_{l,h}=\sum_{t,d_k}\frac{\partial \mathcal L}{\partial y_{l,h}}\cdot y_{l,h}$
   captured by non-invasive PyTorch hooks on each attention output projection.
   The Taylor importance is the token-weighted $|g_{l,h}|$; the Fisher metric is
   $\mathbb E[g_{l,h}^2]$; entropy profiling uses explicit attention maps.
3. **Global excision.** Heads are ranked globally (or per layer) and the
   $\mathrm{round}(L\cdot N_h\cdot P)$ least-important heads receive $\alpha=0$.
4. **Physical surgery** removes the corresponding QKV/O parameter blocks
   (GQA-aware group alignment for SmolLM), verified numerically identical to gating.
5. **Evaluation.** Sliding-window perplexity (`stride=256, max_length=512`),
   LAMBADA target-word completion, decode latency (B ∈ {1,4,8}), KV-cache MB.

## Headline measured results (all machine-generated)

| Finding | Value |
|---------|-------|
| Max head excision under ΔPPL < 5% (no fine-tuning) | **5%** GPT-2 Small · **10%** GPT-2 Medium · **5.56%** SmolLM-135M |
| GPT-2 Small WT-2 PPL at 0% → 30% excision | 36.6757 → 55.6953 (+51.86%) |
| 4-way ablation at 30% (WT-2 PPL) | Taylor **55.70** ≈ Fisher 55.38 ≪ Random 270.78 ± 155.32 ≪ L2-norm 641.17 |
| Layer asymmetry (GPT-2 Small, 30% excision) | Early L0–L3: only 2/48 heads excised (95.8% survival); Deep L8–L11: 22/48 |
| Sensitivity-mass share, GPT-2 Small L0–L3 | 45.2% |
| Decode speedup after 30% physical surgery | +18.2% (B=1), +4.2% (B=4), +11.0% (B=8) |
| KV-cache (Theorem 3, B=8, T=1024, fp16) | 288.00 MB → 192.00 MB at 30% excision (exact 33.3% reduction) |
| Taylor–Fisher head-rank correlation | r = 0.927 (GPT-2 Small) |
| Surgical vs gated logit deviation | < 10⁻⁴ (fp32 roundoff) |

The surprise refuting the pre-registered "intermediate-band" hypothesis: GPT-2
models place near-irreducible heads in the earliest layers (layer 0 mean
sensitivity ≈ 2× network mean), while SmolLM-135M's Pre-LN/RoPE/GQA chassis
produces a U-shaped profile with a defended mid-depth band (51.5% of mass on
L8–L22, only 17.8% of its heads excised at 30%). See `results/summary.json` for
the complete machine-readable digest.

## Repository layout

```
config/      model architecture YAMLs (GPT-2 Small/Medium, SmolLM-135M)
src/core     hooks, model loading, head-slicing metadata, dataset utils
src/metrics  taylor / fisher / entropy / weight-norm importance metrics
src/pruners  global / layerwise / structured (physical surgery) pruners
src/profiler memory tracker, latency bench, FLOPs counter
src/eval     sliding-window perplexity, LAMBADA zero-shot
src/vis      publication figure generators
demo/        Streamlit head-ablation dashboard
tests/       unit tests (hooks, pruners, metrics, eval)
scripts/     experiment runners + table/figure generators
results/     raw JSON telemetry, CSVs, figures
paper/       IEEE manuscript + BibTeX + generated tables + main.pdf
```

## Audit trail

- `results/raw/*.json` — per-window telemetry (token counts, wall-clock seconds)
- `results/*.csv` — flattened artefacts consumed by every figure/table script
- `paper/tables/*.tex` — generated table bodies, embedded verbatim into
  `paper/main.tex` by `scripts/embed_tables.py`
- 4-decimal congruence between `paper/main.tex` Table 1 and
  `results/pruning_curve.csv` is verified by construction
  (`make_tables.py` + `embed_tables.py` chain).

## Citation

```bibtex
@article{rahim2025slmhead,
  title  = {Layer-Wise Attention Head Redundancy and Sensitivity Geometry in
            Sub-Billion Language Models: An Empirical Pruning and Hardware
            Profiling Benchmark},
  author = {Rahim, Md. Mushfiqur},
  year   = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE).
