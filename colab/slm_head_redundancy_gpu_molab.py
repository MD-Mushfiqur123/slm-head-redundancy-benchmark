"""SLM-HEAD-REDUNDANCY-MONUMENT — GPU extension as a marimo notebook.

Run it on molab (https://molab.marimo.io):
  1. New notebook > Add from GitHub:
     MD-Mushfiqur123/slm-head-redundancy-benchmark > colab/slm_head_redundancy_gpu_molab.py
  2. Toggle the GPU in the notebook specs button (header).
  3. Pick a model, press "Run". Artifacts land in results/raw/ inside the
     cloned repo; download the JSON/PNG files below and push them back
     through the maintained pipeline (/TRUTH law: everything measured).
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
    import marimo as mo
    return json, mo, os, subprocess, sys


@app.cell
def _(json, mo, os, subprocess, sys):
    import time
    AI_REPO_URL = "https://github.com/MD-Mushfiqur123/slm-head-redundancy-benchmark"
    AI_REPO = "slm-head-redundancy-benchmark"
    AI_MODEL = mo.ui.dropdown(
        options=[
            "gpt2_small", "gpt2_medium", "smollm_135m",
            "opt_125m", "pythia_160m", "pythia_410m", "qwen2_05b", "all",
        ],
        value="gpt2_small",
        label="model (all = পূর্ণ ৭-মডেল ম্যাট্রিক্স)",
    )
    AI_RATIO = mo.ui.text(value="0.30", label="pruning ratio")
    AI_FP16 = mo.ui.switch(label="fp16 latency/KV profile", value=True)
    AI_FORCE = mo.ui.switch(label="force rerun (ক্যাশ বাদ দিয়ে নতুন করে)", value=False)
    AI_RUN = mo.ui.run_button(
        label="▶️ ALL-IN-ONE: pull → install → run → results → download"
    )
    def _tag(r: float) -> str:
        return ("%.1f" % (100 * r)).replace(".0", "")
    _ui = mo.vstack([
        mo.md(
            "### 🎯 ALL-IN-ONE — এক সেলেই সব\n\n"
            "নিচের বোতামটা চাপলে: ① repo pull ② dependency install "
            "③ পুরো GPU benchmark ④ measured টেবিল ⑤ প্লট ⑥ সব JSON "
            "ডাউনলোড — অন্য কোনো সেলে হাত দিতে হবে না।"
        ),
        AI_MODEL, AI_RATIO, AI_FP16, AI_FORCE, AI_RUN,
    ])
    mo.stop(not AI_RUN.value, _ui)
    _t0 = time.perf_counter()

    if not os.path.isdir(AI_REPO):
        subprocess.run(["git", "clone", "--depth", "1", AI_REPO_URL, AI_REPO], check=True)
    else:
        _pull = subprocess.run(
            ["git", "-C", AI_REPO, "pull", "--ff-only"], capture_output=True, text=True,
        )
        if _pull.returncode != 0:
            raise RuntimeError(f"git pull failed:\n{_pull.stderr[-2000:]}")

    for _p in ("transformers", "datasets", "matplotlib", "tqdm"):
        try:
            __import__(_p)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", _p], check=True)

    ai_model_key = AI_MODEL.value
    ai_ratio = float(AI_RATIO.value)
    _kv_warn = {"smollm_135m": 3, "qwen2_05b": 7}
    if ai_model_key in _kv_warn:
        _g = _kv_warn[ai_model_key]
        _heads = {"smollm_135m": 270, "qwen2_05b": 336}[ai_model_key]
        _n = round(_heads * ai_ratio)
        if _n % _g != 0:
            _alt = (_n // _g) * _g / _heads
            raise ValueError(
                f"{ai_model_key}: ratio {ai_ratio} excises {_n} heads, "
                f"not a multiple of the KV group {_g}; use e.g. {_alt:.6f}"
            )
    ai_cmd = [sys.executable, "scripts/gpu_extension.py", "--model", ai_model_key,
           "--ratio", repr(ai_ratio)]
    if AI_FP16.value:
        ai_cmd.append("--fp16")
    if AI_FORCE.value:
        ai_cmd.append("--force")
    _proc = subprocess.run(
        ai_cmd, capture_output=True, text=True, cwd=os.path.abspath(AI_REPO),
    )
    _log = mo.accordion(
        {"full run log": (_proc.stdout[-20000:] + "\n" + _proc.stderr[-4000:])}
    )
    if _proc.returncode != 0:
        raise RuntimeError(
            f"gpu_extension.py exited {_proc.returncode}\n{_proc.stderr[-2000:]}"
        )
    _mins = (time.perf_counter() - _t0) / 60

    _keys = (
        ["gpt2_small", "gpt2_medium", "smollm_135m", "opt_125m",
         "pythia_160m", "pythia_410m", "qwen2_05b"]
        if ai_model_key == "all" else [ai_model_key]
    )
    _summ = []
    for _k in _keys:
        _p = os.path.join(AI_REPO, "results/raw", f"gpu_summary_{_k}_{_tag(ai_ratio)}.json")
        if os.path.exists(_p):
            _summ.append((_k, json.load(open(_p))))
    _rows = [
        f"| {_k} | {s['gpu']} | {s['excised_heads']}/{s['total_heads']} | "
        f"{s['retention']['wt2_unpruned']['ppl']:.4f} | "
        f"{s['retention']['wt2_pruned']['ppl']:.4f} | "
        f"{100*s['retention']['lambada_unpruned']['top1_accuracy']:.1f}% | "
        f"{100*s['retention']['lambada_pruned']['top1_accuracy']:.1f}% | "
        f"{list(s['kv_cache']['seq_lens'].values())[-1]['reduction_pct']:.1f}% |"
        for _k, s in _summ
    ]
    _table = mo.md(
        "## Measured summary (/TRUTH — runtime-executed)\n\n"
        "| model | GPU | heads excised | WT-2 PPL base | WT-2 PPL pruned "
        "| LAMBADA base | LAMBADA pruned | KV cut |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows)
    ) if _rows else mo.md("আর্টিফ্যাক্ট নেই।")
    _stages = ("summary", "ppl", "latency", "mask", "kv")
    _cards = []
    for _k in _keys:
        for _st in _stages:
            _p = os.path.join(AI_REPO, "results/raw", f"gpu_{_st}_{_k}_{_tag(ai_ratio)}.json")
            if os.path.exists(_p):
                with open(_p, "rb") as _fh:
                    _cards.append(
                        mo.download(data=fh.read(), filename=os.path.basename(_p))
                    )
    _figs = []
    for _k in _keys:
        for _fig in ("latency", "kv"):
            _p = os.path.join(AI_REPO, "results/figures", f"gpu_{_fig}_{_k}.png")
            if os.path.exists(_p):
                _figs.append(mo.image(src=_p, width=720))
    return mo.vstack([
        mo.md(f"## ✅ ALL-IN-ONE শেষ ({_mins:.1f} মিনিট)"),
        _table, out,
        mo.md("### 📥 JSON artifacts")] + _cards
        + [mo.md("### 🖼 figures")] + _figs + [_ui],
    )


@app.cell
def _(os, subprocess):
    REPO_URL = "https://github.com/MD-Mushfiqur123/slm-head-redundancy-benchmark"
    REPO_DIR = "slm-head-redundancy-benchmark"
    if not os.path.isdir(REPO_DIR):
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, REPO_DIR],
            check=True,
        )
    return REPO_DIR,


@app.cell
def _(subprocess, sys):
    def _ensure(pkg: str) -> None:
        try:
            __import__(pkg)
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", pkg], check=True
            )

    for _p in ("transformers", "datasets", "matplotlib", "tqdm"):
        _ensure(_p)
    return


@app.cell
def _(mo, subprocess):
    try:
        _p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        _name = _p.stdout.strip() or "no GPU detected"
    except FileNotFoundError:
        _name = "no GPU detected (nvidia-smi নেই)"
    mo.vstack([
        mo.md(f"**GPU:** {_name}"),
        mo.md("GPU না থাকলে notebook specs বোতাম থেকে GPU অন করো (molab: RTX Pro 6000 Blackwell, 96 GB)।"),
    ])
    return


@app.cell
def _(mo):
    MODEL = mo.ui.dropdown(
        options=[
            "gpt2_small", "gpt2_medium", "smollm_135m",
            "opt_125m", "pythia_160m", "pythia_410m", "qwen2_05b", "all",
        ],
        value="gpt2_small",
        label="model (all = পূর্ণ ৭-মডেল ম্যাট্রিক্স)",
    )
    RATIO = mo.ui.text(value="0.30", label="pruning ratio", full_width=False)
    FP16 = mo.ui.switch(label="fp16 latency/KV profile", value=False)
    FORCE = mo.ui.switch(label="force rerun (ক্যাশ বাদ দিয়ে নতুন করে)", value=False)
    RUN = mo.ui.run_button(label="▶️ Run GPU extension")
    mo.vstack([MODEL, RATIO, FP16, FORCE, RUN])
    return FORCE, FP16, MODEL, RATIO, RUN


@app.cell
def _(FORCE, FP16, MODEL, RATIO, RUN, mo, os, subprocess, sys):
    mo.stop(
        not RUN.value,
        "বোতামে ক্লিক করলে রান শুরু হবে (resumable: এক রানের আর্টিফ্যাক্ট আরেক রানে কাজে লাগে)।",
    )
    model_key = MODEL.value
    ratio = float(RATIO.value)
    _kv_warn = {
        "smollm_135m": 3,
        "qwen2_05b": 7,
    }
    import math

    if model_key in _kv_warn:
        g = _kv_warn[model_key]
        heads = {"smollm_135m": 270, "qwen2_05b": 336}[model_key]
        n = round(heads * ratio)
        if n % g != 0:
            _alt = (n // g) * g / heads
            raise ValueError(
                f"{model_key}: ratio {ratio} excises {n} heads, "
                f"not a multiple of the KV group {g}; use e.g. {_alt:.6f}"
            )
    cmd = [sys.executable, "scripts/gpu_extension.py", "--model", model_key,
           "--ratio", repr(ratio)]
    if FP16.value:
        cmd.append("--fp16")
    if FORCE.value:
        cmd.append("--force")
    _proc = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=os.path.abspath("slm-head-redundancy-benchmark"),
    )
    out = mo.accordion(
        {"full run log": (_proc.stdout[-20000:] + "\n" + _proc.stderr[-4000:])}
    )
    if _proc.returncode != 0:
        raise RuntimeError(f"gpu_extension.py exited {_proc.returncode}")
    mo.vstack([
        mo.md("## ✅ রান শেষ — নিচে ফলাফল ও ডাউনলোড"),
        out,
    ])
    return model_key, ratio


@app.cell
def _(json, mo, os, ratio, model_key):
    def _tag(r: float) -> str:
        return ("%.1f" % (100 * r)).replace(".0", "")

    _keys = (
        ["gpt2_small", "gpt2_medium", "smollm_135m", "opt_125m",
         "pythia_160m", "pythia_410m", "qwen2_05b"]
        if model_key == "all"
        else [model_key]
    )
    _summaries = []
    for _k in _keys:
        _p = os.path.join(
            "slm-head-redundancy-benchmark",
            "results/raw", f"gpu_summary_{_k}_{_tag(ratio)}.json",
        )
        if os.path.exists(_p):
            _summaries.append((_k, _p, json.load(open(_p))))
    _rows = [
        f"| {_k} | {s['gpu']} | {s['excised_heads']}/{s['total_heads']} | "
        f"{s['retention']['wt2_unpruned']['ppl']:.4f} | "
        f"{s['retention']['wt2_pruned']['ppl']:.4f} | "
        f"{100*s['retention']['lambada_unpruned']['top1_accuracy']:.1f}% | "
        f"{100*s['retention']['lambada_pruned']['top1_accuracy']:.1f}% | "
        f"{list(s['kv_cache']['seq_lens'].values())[-1]['reduction_pct']:.1f}% |"
        for _k, _p, s in _summaries
    ]
    mo.md(
        "## Measured summary (/TRUTH — runtime-executed)\n\n"
        "| model | GPU | heads excised | WT-2 PPL base | WT-2 PPL pruned | LAMBADA base | LAMBADA pruned | KV cut |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(_rows)
    ) if _rows else mo.md("আর্টিফ্যাক্ট নেই — আগে রান করো।")
    return (os,)


@app.cell
def _(json, mo, os, ratio, model_key):
    def _tag2(r: float) -> str:
        return ("%.1f" % (100 * r)).replace(".0", "")

    _keys2 = (
        ["gpt2_small", "gpt2_medium", "smollm_135m", "opt_125m",
         "pythia_160m", "pythia_410m", "qwen2_05b"]
        if model_key == "all"
        else [model_key]
    )
    _stages = ("summary", "ppl", "latency", "mask", "kv")
    _files = []
    for _k in _keys2:
        for _st in _stages:
            _p = os.path.join(
                "slm-head-redundancy-benchmark",
                "results/raw", f"gpu_{_st}_{_k}_{_tag2(ratio)}.json",
            )
            if os.path.exists(_p):
                _files.append(_p)
    _cards = []
    for _p in _files:
        with open(_p, "rb") as fh:
            _cards.append(
                mo.download(data=fh.read(), filename=os.path.basename(_p))
            )
    mo.vstack(
        [mo.md("### 📥 download measured artifacts (JSON)")] + _cards
        if _cards else [mo.md("কোনো JSON আর্টিফ্যাক্ট নেই।")]
    )
    return


@app.cell
def _(mo, os, ratio, model_key):
    def _tag3(r: float) -> str:
        return ("%.1f" % (100 * r)).replace(".0", "")

    _keys3 = (
        ["gpt2_small", "gpt2_medium", "smollm_135m", "opt_125m",
         "pythia_160m", "pythia_410m", "qwen2_05b"]
        if model_key == "all"
        else [model_key]
    )
    _figs = []
    for _k in _keys3:
        for _fig in ("latency", "kv"):
            _p = os.path.join(
                "slm-head-redundancy-benchmark",
                "results/figures", f"gpu_{_fig}_{_k}.png",
            )
            if os.path.exists(_p):
                _figs.append(mo.image(src=_p, width=720))
    mo.vstack(
        [mo.md("### 🖼 measured figures")] + _figs
        if _figs else [mo.md("প্লট নেই (--no-plots এড়ানো হয়েছে বা রান হয়নি)।")]
    )
    return


@app.cell
def _(mo):
    FT_MODEL = mo.ui.dropdown(
        options=[
            "gpt2_small", "gpt2_medium", "smollm_135m",
            "opt_125m", "pythia_160m", "pythia_410m", "qwen2_05b",
        ],
        value="gpt2_small",
        label="repayment model",
    )
    FT_RATIOS = mo.ui.text(
        value="0.1 0.2 0.3 0.4",
        label="excision ratios (space-separated; GQA models auto-snap)",
    )
    FT_EPOCHS = mo.ui.number(value=2, start=1, stop=10, label="fine-tune epochs")
    FT_WINDOWS = mo.ui.number(
        value=2000, start=100, step=100, label="train windows (512 tok each)")
    FT_LR = mo.ui.text(value="1e-5", label="AdamW lr")
    FT_RUN = mo.ui.run_button(label="▶️ Run C — fine-tuning repayment (~30–90 min)")
    mo.vstack([mo.md(
        "### 💰 Fine-tuning repayment\n\n"
        "GQA-safe surgery → AdamW fine-tune → pre/post-repayment PPL "
        "+ repair-% per ratio. Everything measured at runtime (/TRUTH)."
    ), FT_MODEL, FT_RATIOS, FT_EPOCHS, FT_WINDOWS, FT_LR, FT_RUN])
    return FT_EPOCHS, FT_LR, FT_MODEL, FT_RUN, FT_RATIOS, FT_WINDOWS


@app.cell
def _(
    FT_EPOCHS, FT_LR, FT_MODEL, FT_RUN, FT_RATIOS, FT_WINDOWS, mo, os, subprocess, sys,
):
    mo.stop(
        not FT_RUN.value,
        "Repayment-এর বোতামে ক্লিক করলে রান শুরু হবে "
        "(ratios পেইন, ফাইনটাইন, ঘণ্টাঘণ্টি লেগে)।",
    )
    ft_model_key = FT_MODEL.value
    _ratios = " ".join(FT_RATIOS.value.split()) or "0.3"
    try:
        _lr = str(float(FT_LR.value))
    except ValueError:
        raise ValueError(f"--lr must parse as float, got {FT_LR.value!r}")
    _cmd = [
        sys.executable, "scripts/finetune_repayment.py",
        "--model", ft_model_key,
        "--ratios", *_ratios.split(),
        "--epochs", str(int(FT_EPOCHS.value)),
        "--train-windows", str(int(FT_WINDOWS.value)),
        "--lr", _lr,
    ]
    _proc = subprocess.run(
        _cmd, capture_output=True, text=True,
        cwd=os.path.abspath("slm-head-redundancy-benchmark"),
    )
    _log = mo.accordion(
        {"full repayment log": (_proc.stdout[-20000:] + "\n" + _proc.stderr[-4000:])}
    )
    if _proc.returncode != 0:
        raise RuntimeError(
            f"finetune_repayment.py exited {_proc.returncode}\n{_proc.stderr[-2000:]}"
        )
    mo.vstack([mo.md("## ✅ পুনঃশোধন রান শেষ — নিচে ফলাফল ও ডাউনলোড"), _log])
    return (ft_model_key,)


@app.cell
def _(json, mo, os):
    _repo = "slm-head-redundancy-benchmark"
    _raw = os.path.join(_repo, "results", "raw")
    _jsons = sorted(
        f for f in os.listdir(_raw)
        if f.startswith("repayment_") and f.endswith(".json")
    ) if os.path.isdir(_raw) else []
    _mds = []
    for _name in _jsons:
        with open(os.path.join(_raw, _name)) as _fh:
            _res = json.load(_fh)
        _tbl = (
            "| ratio | pre-repayment PPL | post-repayment PPL | ΔPPL after | repair % | FT seconds |\n"
            "|---|---|---|---|---|---|\n"
            + "\n".join(
                f"| {100*r['ratio']:.0f}% | {r['pre_ppl']:.4f} | {r['post_ppl']:.4f} "
                f"| {r['post_delta_pct']:+.2f}% | {r['repair_pct']:.1f}% "
                f"| {r['ft_seconds']:.1f} |"
                for r in _res["rows"]
            )
        )
        _mds.append(
            mo.md(
                f"## 📊 Repayment table — {_res['model']}\n\n"
                f"unpruned PPL (base): {_res['unpruned_ppl']:.4f}\n\n{_tbl}"
            )
        )
    _ = mo.vstack(
        _mds if _mds else [mo.md("পুনঃশোধনের আর্টিফ্যাক্ট নেই — আগে রান করো।")]
    )
    return


@app.cell
def _(mo, os):
    _repo = "slm-head-redundancy-benchmark"
    _raw = os.path.join(_repo, "results", "raw")
    _files = []
    for _root in (_raw, _repo):
        if not os.path.isdir(_root):
            continue
        for _name in sorted(
            f for f in os.listdir(_root)
            if f.startswith("repayment_")
            and (f.endswith(".json") or f.endswith(".csv"))
        ):
            _files.append(os.path.join(_root, _name))
    _cards = []
    for _p in _files:
        with open(_p, "rb") as _fh:
            _cards.append(
                mo.download(data=_fh.read(), filename=os.path.basename(_p))
            )
    _header = mo.md("### 📥 download repayment artifacts (সব মডেল)")
    _empty = mo.md("\u0995\u09cb\u09a8\u09cb artifact \u09a8\u09c7\u07a5\u0965")
    _ = mo.vstack([_header] + _cards if _cards else [_empty])
    return


if __name__ == "__main__":
    app.run()
