"""SLM-HEAD-REDUNDANCY-MONUMENT — LONG-JOBS notebook (marimo).

The two multi-hour experiments that complete the paper's measurement matrix,
on one molab / 12-hour GPU session (https://molab.marimo.io):

  RUN B — 7-architecture matrix: `scripts/gpu_extension.py --model all`
          (resumable: per-stage JSON caches; GQA ratios auto-snap).
  RUN C — fine-tuning repayment: `scripts/finetune_repayment.py`
          (GQA-safe surgery -> AdamW -> pre/post PPL + repair-% per ratio).

Add it on molab:  New notebook > Add from GitHub >
  MD-Mushfiqur123/slm-head-redundancy-benchmark > colab/slm_head_redundancy_long_molab.py
Toggle the GPU in the specs button, press a Run button, and download the
artifacts at the bottom. /TRUTH law: every value is measured at runtime.
"""

import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    import time

    import marimo as mo
    return json, mo, os, subprocess, sys, time


@app.cell
def _(os, subprocess):
    REPO_URL = "https://github.com/MD-Mushfiqur123/slm-head-redundancy-benchmark"
    REPO_DIR = "slm-head-redundancy-benchmark"
    if os.path.isdir(REPO_DIR):
        subprocess.run(["git", "pull", "--ff-only"], cwd=REPO_DIR, check=False)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
            check=True,
        )
    return (REPO_DIR,)


@app.cell
def _(subprocess, sys):
    _procs = [
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            check=False,
        )
        for pkg in ("transformers", "datasets", "matplotlib", "tqdm")
    ]
    return


@app.cell
def _(mo, subprocess):
    try:
        _gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=20,
        )
        _info = _gpu.stdout.strip() if _gpu.returncode == 0 else "nvidia-smi missing"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _info = "nvidia-smi missing — GPU স্পেক্স বাটন থেকে ডিআলোগ করো"
    _ = mo.md(f"### 🎛 GPU: `{_info}`")
    return


@app.cell
def _(mo):
    mo.md(
        "# 🕐 LONG-JOBS notebook\n"
        "দুটো multi-hour পরীক্ষা:\n\n"
        "| Run | কী করে | সময় |\n"
        "|---|---|---|\n"
        "| **Run B** | ৭-আর্কিটেকচার matrix (`--model all`, resumable) | ~২–৩ ঘণ্টা |\n"
        "| **Run C** | fine-tuning repayment (প্রতি ratio pre/post PPL + repair %) | ~৩০–৯০ মিনিট |\n\n"
        "দুটোই `/TRUTH` পাইপলাইন — প্রতিটা সংখ্যা runtime-এ মাপা, কোনো mock নেই।"
    )
    return


@app.cell
def _(mo):
    mo.md("---")
    return


@app.cell
def _(mo):
    mo.md("## 💰 Run C — fine-tuning repayment\n\n"
          "GQA-safe surgery → AdamW fine-tune → **repair %** প্রতি ratio-তে। "
          "WikiText-2 train split (eval-এর সাথে disjoint) বা নিজের corpus ব্যবহার করতে পারো।")
    return


@app.cell
def _(mo):
    FT_MODEL = mo.ui.dropdown(
        options=[
            "gpt2_small", "gpt2_medium", "smollm_135m",
            "opt_125m", "pythia_160m", "pythia_410m", "qwen2_05b",
        ],
        value="gpt2_small",
        label="repayment model (GQA হলে ratio auto-snap হবে)",
    )
    FT_RATIOS = mo.ui.text(
        value="0.1 0.2 0.3 0.4",
        label="excision ratios (space-separated)",
    )
    FT_EPOCHS = mo.ui.number(value=2, start=1, stop=10, label="fine-tune epochs")
    FT_WINDOWS = mo.ui.number(
        value=2000, start=100, step=100, label="train windows (512 tok each)")
    FT_LR = mo.ui.text(value="1e-5", label="AdamW lr")
    FT_CORPUS = mo.ui.text(
        value="",
        label="train corpus path (খালি = wikitext-2 train split; .txt/.csv/.jsonl/hf:org/name)",
        full_width=True,
    )
    FT_RUN = mo.ui.run_button(label="▶️ Run C — repayment (~30–90 min)")
    mo.vstack([FT_MODEL, FT_RATIOS, FT_EPOCHS, FT_WINDOWS, FT_LR, FT_CORPUS, FT_RUN])
    return FT_CORPUS, FT_EPOCHS, FT_LR, FT_MODEL, FT_RUN, FT_RATIOS, FT_WINDOWS


@app.cell
def _(
    FT_CORPUS, FT_EPOCHS, FT_LR, FT_MODEL, FT_RUN, FT_RATIOS, FT_WINDOWS,
    REPO_DIR, mo, os, subprocess, sys, time,
):
    mo.stop(
        not FT_RUN.value,
        "Run C বোতামে ক্লিক করলে শুরু হবে (ratios সংখ্যা মতো লম্বা লাগবে)।",
    )
    ft_model_key = FT_MODEL.value
    _ratios = " ".join(FT_RATIOS.value.split()) or "0.3"
    try:
        _lr = str(float(FT_LR.value))
    except ValueError:
        raise ValueError(f"--lr float হতে হবে, পেয়েছি {FT_LR.value!r}")
    _cmd = [
        sys.executable, "scripts/finetune_repayment.py",
        "--model", ft_model_key,
        "--ratios", *_ratios.split(),
        "--epochs", str(int(FT_EPOCHS.value)),
        "--train-windows", str(int(FT_WINDOWS.value)),
        "--lr", _lr,
    ]
    if FT_CORPUS.value.strip():
        _cmd += ["--corpus", FT_CORPUS.value.strip()]
    _t0 = time.perf_counter()
    _proc = subprocess.run(
        _cmd, capture_output=True, text=True,
        cwd=os.path.abspath(REPO_DIR),
    )
    _mins = (time.perf_counter() - _t0) / 60
    _log = mo.accordion(
        {"full repayment log": (_proc.stdout[-20000:] + "\n" + _proc.stderr[-4000:])}
    )
    if _proc.returncode != 0:
        raise RuntimeError(
            f"finetune_repayment.py exited {_proc.returncode}\n{_proc.stderr[-2000:]}"
        )
    mo.vstack([mo.md(f"## ✅ Run C শেষ ({_mins:.1f} মিনিট) — নিচে ফলাফল ও ডাউনলোড"), _log])
    return (ft_model_key,)


@app.cell
def _(json, mo, os, REPO_DIR, ft_model_key):
    ft_json_path = os.path.join(REPO_DIR, "results/raw", f"repayment_{ft_model_key}.json")
    ft_csv_path = os.path.join(REPO_DIR, f"repayment_{ft_model_key}.csv")
    if os.path.exists(ft_json_path):
        _res = json.load(open(ft_json_path))
        _rows = [
            f"| {100 * r['ratio']:.0f}% | {r['pre_ppl']:.4f} | {r['post_ppl']:.4f} "
            f"| {r['post_delta_pct']:+.2f}% | {r['repair_pct']:.1f}% "
            f"| {r['ft_seconds']:.1f} |"
            for r in _res["rows"]
        ]
        _ = mo.md(
            f"### 📊 Repayment table — {ft_model_key}\n\n"
            f"unpruned PPL (base): {_res['unpruned_ppl']:.4f}\n\n"
            "| ratio | pre-repayment PPL | post-repayment PPL | ΔPPL after | repair % | FT seconds |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(_rows)
        )
    else:
        _ = mo.md("আর্টিফ্যাক্ট নেই — আগে Run C করো।")
    return ft_csv_path, ft_json_path


@app.cell
def _(ft_csv_path, ft_json_path, mo, os):
    _cards = []
    for _p in (ft_json_path, ft_csv_path):
        if os.path.exists(_p):
            with open(_p, "rb") as _fh:
                _cards.append(
                    mo.download(data=_fh.read(), filename=os.path.basename(_p))
                )
    _ = mo.vstack(
        [mo.md("### 📥 Run C artifacts")] + _cards
        if _cards else [mo.md("কোনো artifact নেই।")]
    )
    return


@app.cell
def _(mo):
    mo.md("---")
    return


@app.cell
def _(mo):
    mo.md(
        "## 🏛 Run B — ৭-আর্কিটেকচার matrix\n\n"
        "`scripts/gpu_extension.py --model all` — GPT-2 Small/Medium, SmolLM-135M,\n"
        "OPT-125M, Pythia-160m/410m, Qwen2.5-0.5B; প্রতি স্টেজ JSON-এ ক্যাশ হয়,\n"
        "তাই মাঝপথে বন্ধ হলে আবার চাপলে বাকিটা ধরবে। Qwen2.5-0.5B-র ratio নিজেই\n"
        "0.30 → 0.291667-এ স্ন্যাপ হবে।"
    )
    return


@app.cell
def _(mo):
    B_FP16 = mo.ui.switch(label="fp16 protocol (মডেল ক্যালিব্রেশনের আগেই half হবে)", value=False)
    B_RATIO = mo.ui.text(value="0.30", label="ratio (GQA autosnap আছে)")
    B_BATCH = mo.ui.number(value=8, start=1, stop=64, label="batch size")
    B_RUN = mo.ui.run_button(label="▶️ Run B — full 7-model matrix (~২–৩ ঘণ্টা)")
    mo.vstack([B_FP16, B_RATIO, B_BATCH, B_RUN])
    return B_BATCH, B_FP16, B_RATIO, B_RUN


@app.cell
def _(B_BATCH, B_FP16, B_RATIO, B_RUN, REPO_DIR, mo, os, subprocess, sys, time):
    mo.stop(
        not B_RUN.value,
        "Run B বোতামে ক্লিক করলে ৭-মডেল matrix চলবে (resumable)।",
    )
    _ratio = float(B_RATIO.value)
    _cmd = [
        sys.executable, "scripts/gpu_extension.py", "--model", "all",
        "--ratio", repr(_ratio), "--batch-size", str(int(B_BATCH.value)),
    ]
    if B_FP16.value:
        _cmd.append("--fp16")
    _t0 = time.perf_counter()
    _proc = subprocess.run(
        _cmd, capture_output=True, text=True,
        cwd=os.path.abspath(REPO_DIR),
    )
    _hrs = (time.perf_counter() - _t0) / 3600
    _log = mo.accordion(
        {"full matrix log": (_proc.stdout[-25000:] + "\n" + _proc.stderr[-4000:])}
    )
    if _proc.returncode != 0:
        raise RuntimeError(
            f"gpu_extension.py exited {_proc.returncode}\n{_proc.stderr[-2000:]}"
        )
    mo.vstack([mo.md(f"## ✅ Run B শেষ ({_hrs:.2f} ঘণ্টা)"), _log])
    return


@app.cell
def _(json, mo, os, REPO_DIR):
    _rows = []
    runb_files = sorted(
        f for f in os.listdir(os.path.join(REPO_DIR, "results/raw"))
        if f.startswith("gpu_summary_")
    )
    for _name in runb_files:
        try:
            _s = json.load(open(os.path.join(REPO_DIR, "results/raw", _name)))
            r = dict(_s["retention"])
            _rows.append(
                f"| {_s['model']} | {_s['gpu'][:28]} | {_s['excised_heads']}/{_s['total_heads']} "
                f"| {r['wt2_unpruned']['ppl']:.3f} | {r['wt2_pruned']['ppl']:.3f} "
                f"| {r['ptb_pruned']['ppl']:.3f} "
                f"| {100 * r['lambada_unpruned']['top1_accuracy']:.1f}% "
                f"| {100 * r['lambada_pruned']['top1_accuracy']:.1f}% |"
            )
        except (KeyError, json.JSONDecodeError):
            continue
    _ = mo.md(
        "### 📊 7-model measured summary\n\n"
        "| model | GPU | heads | WT2 base | WT2 pruned | PTB pruned | LAMBADA base | LAMBADA pruned |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows)
        if _rows else
        "এখনো কোনো summary নেই — আগে Run B করো।"
    )
    return runb_files


@app.cell
def _(runb_files, mo, os, REPO_DIR):
    _cards = []
    for _name in runb_files:
        for _st in ("ppl", "latency", "mask", "kv"):
            _p = os.path.join(REPO_DIR, "results/raw",
                              _name.replace("gpu_summary_", f"gpu_{_st}_"))
            if os.path.exists(_p):
                with open(_p, "rb") as _fh:
                    _cards.append(
                        mo.download(data=_fh.read(), filename=os.path.basename(_p))
                    )
    _ = mo.vstack(
        [mo.md("### 📥 Run B artifacts")] + _cards
        if _cards else [mo.md("কোনো artifact নেই।")]
    )
    return


@app.cell
def _(mo):
    mo.md(
        "---\n"
        "### 🤝 শেষ ধাপ\n"
        "ডাউনলোড করা JSON/CSV গুলো ভাইব (Vibe) সেশনে আপলোড করো — কমিট + পেপারে\n"
        "বসানো + PDF rebuild আমি করব (৪-দশমিক audit trail সহ, `/TRUTH`)।"
    )
    return


if __name__ == "__main__":
    app.run()
