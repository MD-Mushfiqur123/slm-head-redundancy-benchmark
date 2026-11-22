# SLM-HEAD-REDUNDANCY-MONUMENT

**Layer-Wise Attention Head Redundancy and Sensitivity Geometry in Sub-Billion Language Models: An Empirical Pruning and Hardware Profiling Benchmark**

Author: **Md. Mushfiqur Rahim** — Department of Science, Govt. Azizul Haque College, Bogura, Bangladesh.

This repository is the complete, reproducible implementation of an empirical study of
attention-head redundancy in **seven** sub-billion-parameter language models (GPT-2 Small
124M, GPT-2 Medium 355M, SmolLM-135M, OPT-125M, Pythia-160M, Pythia-410M, Qwen2.5-0.5B):

- exact-gradient **first-order Taylor** head-sensitivity scoring,
- **empirical Fisher** (second-order) head scoring,
- calibration-free **L2 weight-norm** and **random** control baselines,
- **zero-shot structural head excision** (no fine-tuning, no calibration data at inference),
- verification that physical weight surgery is *bit-equivalent* to gated ablation,
- sliding-window perplexity (WikiText-2 + Penn Treebank), zero-shot LAMBADA completion,
- wall-clock decode-latency benchmarks and Theorem-3 KV-cache memory accounting,
- publication figures and an 8-page IEEE manuscript (`paper/main.tex`, compiled copy at
  the repo root),
- a **paper-evidence notebook** (`colab/paper_evidence.py`) that verifies every paper
  claim against the measured raw artifacts in seconds.

**[Read the paper (PDF)](https://github.com/MD-Mushfiqur123/slm-head-redundancy-benchmark/blob/main/SLM-HEAD-REDUNDANCY-MONUMENT.pdf)**

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m unittest discover tests/          # unit tests (48 tests)
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

Interactive exploration: `streamlit run demo/app.py`.

## Paper-evidence notebook (no GPU required)

`colab/paper_evidence.py` verifies every claim of the manuscript in seconds, directly
against the measured artifacts in `results/raw/`: a no-mock audit of all raw JSONs,
sweep-matrix completeness, pruning-mask geometry, surgery-vs-gated-ablation logit
equivalence, retention tables, Theorem-3 KV envelope, latency tables, a cross-check that
the paper's exact numbers appear in `paper/main.tex`, the full figure gallery, and a
one-shot bundle download. Open it at:

```
https://molab.marimo.io/github/MD-Mushfiqur123/slm-head-redundancy-benchmark/blob/main/colab/paper_evidence.py
```

## GPU measurement notebooks (molab)

The CPU-only development environment produced no wall-clock telemetry; the GPU
measurement campaign ran on **molab (marimo)** with an **NVIDIA RTX Pro 6000 Blackwell
Server Edition (96 GB)**, fp16, torch 2.11.0. Any notebook opens by mirroring its GitHub
URL (replace `github.com` with `molab.marimo.io/github`):

| Notebook | Purpose |
|----------|---------|
| `colab/slm_head_redundancy_gpu_blackwell.py` | benchmark / fine-tuning repayment for a chosen model (CUDA calibration, latency, KV) |
| `colab/slm_head_redundancy_long_molab.py` | 12-hour session: full 7-architecture matrix + repayment sweep |
| `colab/sweep_all_models.py` | full-ratio sweep (10/20/40/50/60%), resumable, with bundle download |
| `colab/slm_head_redundancy_gpu.ipynb` | original Colab T4 variant (historical) |

All notebooks drive the same measured pipeline (`scripts/gpu_extension.py`, GQA-safe
ratio snapping included) and offer every raw JSON / CSV / figure for download.
Canonical 30%/29.2% ratio artifacts are never overwritten.

## GPU CLI

```bash
python scripts/gpu_extension.py --model gpt2_small --ratio 0.30
python scripts/gpu_extension.py --model smollm_135m --ratio 0.055556
# user corpus instead of WikiText-2 for calibration + WT2-style eval:
python scripts/gpu_extension.py --model opt_125m --ratio 0.30 --corpus data/my_corpus.txt
```

The registry covers **seven sub-billion architectures**, spanning four parameter
families: GPT-2 Small/Medium (Post-LN, absolute positions), **SmolLM-135M** (LLaMA
family, RoPE, SwiGLU, GQA group 3), **OPT-125M** (pre-LN, learned positions),
**Pythia-160M/410M** (GPT-NeoX, RoPE, fused interleaved QKV with dedicated physical
surgery), and **Qwen2.5-0.5B** (RoPE, SwiGLU, GQA group 7). GQA models take a
group-aligned ratio (e.g. `--ratio 0.291667` for Qwen2.5-0.5B; 0.30 for SmolLM). The
whole matrix runs sequentially with `--model all`; per-stage caches in `results/raw/`
make the runs resumable (`--force` recomputes).

### Fine-tuning repayment experiment (the paper's Future-Work track i)

`scripts/finetune_repayment.py` measures how much of the excision damage a small
fine-tuning budget repairs:

```bash
python scripts/finetune_repayment.py --model gpt2_small \
    --ratios 0.1 0.2 0.3 0.4 --epochs 2 --train-windows 2000
# with a user corpus (calib+eval) and a disjoint training corpus:
python scripts/finetune_repayment.py --model gpt2_small \
    --ratios 0.3 --corpus data/eval_corpus.txt --train-corpus data/train_corpus.txt
```

It calibrates Taylor importance exactly like the main pipeline, excises with GQA-safe
physical surgery, fine-tunes each pruned model (AdamW, cosine schedule) and writes
pre/post-repayment perplexity, ΔPPL and repair-% per ratio to
`results/raw/repayment_<model>.json` + `results/repayment_<model>.csv`.

### User-supplied corpora

`src/core/custom_dataset.py` loads user datasets for any pipeline stage via the
`--corpus` flag: `.txt` (one document per line), `.csv`/`.tsv` (text column
auto-detected or forced with `text_column=`), `.jsonl` (first of
`text/content/body/sentence/document`), or `hf:org/name` Hugging Face datasets.
The first `--num-calibration` windows go to calibration, the next `--num-eval` to
evaluation, preserving the paper's zero-overlap isolation protocol.

## Measured full-ratio sweep inventory (RTX PRO 6000 Blackwell, fp16)

`results/raw/` carries, for each of the six multi-head models and each ratio tag
10/20/40/50/60 (plus the canonical 30%/29.2% points):

- `gpu_ppl_<model>_<tag>.json` — WT-2 / PTB perplexity + LAMBADA (pruned and unpruned),
- `gpu_mask_<model>_<tag>.json` — the excised-head mask,
- `gpu_kv_<model>_<tag>.json` — live KV-cache bytes vs Theorem 3,
- `gpu_latency_<model>_<tag>.json` — decode latency at B ∈ {1,4,8},
- `gpu_summary_<model>_<tag>.json` — the combined stage summary.

Paper Table `tab:fullratio` (compiled into `paper/tables/table9.tex`) reports the
measured WT-2 perplexity matrix. The sweep is complete: every model × tag carries all
five artifact types.

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
   When every head of a layer is ablated, the o-projection weight is zeroed but the
   shared output bias is retained, matching the gated model's $o(0)=b$ semantics.
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
| Full-ratio sweep, Blackwell fp16 (Tab. fullratio) | ΔPPL < 5% budget survives only the 10% point (4/6 models; none for GPT-2 Small and SmolLM-135M) |

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
