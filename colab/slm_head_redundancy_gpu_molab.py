# SLM-HEAD-REDUNDANCY-MONUMENT — GPU benchmark (ALL-IN-ONE)
#
# একটাই অপারেশন সেল, একটাই বোতাম:
#   ▶️ ALL-IN-ONE: pull → install → run → tables → figures → download
#
# task = benchmark → scripts/gpu_extension.py (সেভেন-মডেল সাপোর্ট)
# task = repayment → scripts/finetune_repayment.py (fine-tuning repayment)
# task = both      → দুটোই পরপর
#
# /TRUTH: every value measured at runtime — no mocks, no placeholders.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


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
def _(mo):
    mo.md(
        """
        # 🏛️ SLM-HEAD-REDUNDANCY-MONUMENT — GPU ALL-IN-ONE

        **একটাই অপারেশন সেল, একটাই বোতাম।** বোতাম চাপলে: ① repo pull ② dependency
        install ③ benchmark/repayment ④ measured টেবিল ⑤ প্লট ⑥ সব JSON ডাউনলোড।

        /TRUTH LAW: প্রতিটা সংখ্যা runtime-এ মাপা — কোনো mock/placeholder নেই।
        """
    )
    return


@app.cell
def _(mo):
    AI_MODELS = [
        "gpt2_small", "gpt2_medium", "smollm_135m",
        "opt_125m", "pythia_160m", "pythia_410m", "qwen2_05b", "all",
    ]
    AI_TASK = mo.ui.dropdown(
        options=["benchmark", "repayment", "both"],
        value="benchmark",
        label="task",
    )
    AI_MODEL = mo.ui.dropdown(
        options=AI_MODELS,
        value="gpt2_small",
        label="model (all = পূর্ণ ৭-মডেল ম্যাট্রিক্স)",
    )
    AI_RATIO = mo.ui.text(value="0.30", label="pruning ratio")
    AI_FP16 = mo.ui.switch(label="fp16 latency/KV profile", value=True)
    AI_FORCE = mo.ui.switch(label="force rerun (ক্যাশ বাদ দিয়ে নতুন করে)", value=False)
    AI_RUN = mo.ui.run_button(
        label="▶️ ALL-IN-ONE: pull → install → run → tables → download"
    )
    AI_UI = mo.vstack([
        mo.md(
            "### 🎯 ALL-IN-ONE\n\n"
            "setting বদলে বোতামে চাপ দাও — বাকি সব এক সেলেই।"
        ),
        AI_TASK, AI_MODEL, AI_RATIO, AI_FP16, AI_FORCE, AI_RUN,
    ])
    return AI_FP16, AI_FORCE, AI_MODELS, AI_MODEL, AI_RATIO, AI_RUN, AI_TASK, AI_UI


@app.cell
def _(
    AI_FP16,
    AI_FORCE,
    AI_MODEL,
    AI_MODELS,
    AI_RATIO,
    AI_RUN,
    AI_TASK,
    AI_UI,
    json,
    mo,
    os,
    subprocess,
    sys,
    time,
):
    _REPO_URL = "https://github.com/MD-Mushfiqur123/slm-head-redundancy-benchmark"
    _REPO = "slm-head-redundancy-benchmark"
    mo.stop(
        not AI_RUN.value,
        mo.vstack([
            mo.md(
                "⏸️ ALL-IN-ONE এখনো চালু হয়নি — উপরের বোতামে চাপ দাও।\n\n"
                "(setting বদলালে এই সেল auto re-run হয়, কিছু ভারী হয় না।)"
            ),
            AI_UI,
        ]),
    )
    _t0 = time.perf_counter()


    def _banner(text: str) -> None:
        print("─" * 64)
        print(text)
        print("─" * 64)


    if not os.path.isdir(_REPO):
        _banner("① cloning repo")
        subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, _REPO], check=True
        )
    else:
        _banner("① pulling latest repo")
        _pull = subprocess.run(
            ["git", "-C", _REPO, "pull", "--ff-only"],
            capture_output=True, text=True,
        )
        if _pull.returncode != 0:
            raise RuntimeError(f"git pull failed:\n{_pull.stderr[-2000:]}")

    for _pkg in ("transformers", "datasets", "matplotlib", "tqdm"):
        try:
            __import__(_pkg)
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", _pkg], check=True
            )

    task_key = AI_TASK.value
    model_key = AI_MODEL.value
    ratio_val = float(AI_RATIO.value)
    _kv_group = {"smollm_135m": 3, "qwen2_05b": 7}
    _total_heads = {"smollm_135m": 270, "qwen2_05b": 336}
    if model_key in _kv_group:
        _g = _kv_group[model_key]
        _n = round(_total_heads[model_key] * ratio_val)
        if _n % _g != 0:
            _alt = (_n // _g) * _g / _total_heads[model_key]
            raise ValueError(
                f"{model_key}: ratio {ratio_val} excises {_n} heads, "
                f"not a multiple of the KV group {_g}; use e.g. {_alt:.6f}"
            )

    runs = []
    if task_key in ("benchmark", "both"):
        runs.append(
            ("benchmark", [
                sys.executable, "scripts/gpu_extension.py",
                "--model", model_key, "--ratio", repr(ratio_val),
            ] + (["--fp16"] if AI_FP16.value else [])
              + (["--force"] if AI_FORCE.value else []))
        )
    if task_key in ("repayment", "both"):
        if model_key == "all":
            raise ValueError("repayment task: 'all' নয় — একটা মডেল বাছো")
        runs.append(
            ("repayment", [
                sys.executable, "scripts/finetune_repayment.py",
                "--model", model_key,
            ] + (["--force"] if AI_FORCE.value else []))
        )


    def _tag(r: float) -> str:
        return ("%.1f" % (100 * r)).replace(".0", "")


    _tails = {}
    for _task_name, _cmd in runs:
        _banner(f"② running: {_task_name} | {model_key}")
        _proc = subprocess.run(
            _cmd, capture_output=True, text=True, cwd=os.path.abspath(_REPO),
        )
        _tails[_task_name] = _proc.stdout[-20000:] + "\n" + _proc.stderr[-4000:]
        if _proc.returncode != 0:
            raise RuntimeError(
                f"{_task_name} script exited {_proc.returncode}\n"
                f"{_proc.stderr[-2000:]}"
            )
    mins = (time.perf_counter() - _t0) / 60

    key_list = AI_MODELS[:-1] if model_key == "all" else [model_key]
    _raw = os.path.join(_REPO, "results", "raw")
    _fig_dir = os.path.join(_REPO, "results", "figures")


    TABLES = []
    for _k in key_list:
        _p = os.path.join(_raw, f"gpu_summary_{_k}_{_tag(ratio_val)}.json")
        if not os.path.exists(_p):
            continue
        _s = json.load(open(_p))
        TABLES.append(
            mo.md(
                f"### 📊 {_k} — measured summary (/TRUTH, runtime)\n\n"
                "| GPU | heads excised | WT-2 PPL base | WT-2 PPL pruned "
                "| LAMBADA base | LAMBADA pruned | KV cut |\n"
                "|---|---|---|---|---|---|---|\n"
                f"| {_s['gpu']} | {_s['excised_heads']}/{_s['total_heads']} | "
                f"{_s['retention']['wt2_unpruned']['ppl']:.4f} | "
                f"{_s['retention']['wt2_pruned']['ppl']:.4f} | "
                f"{100 * _s['retention']['lambada_unpruned']['top1_accuracy']:.1f}% | "
                f"{100 * _s['retention']['lambada_pruned']['top1_accuracy']:.1f}% | "
                f"{list(_s['kv_cache']['seq_lens'].values())[-1]['reduction_pct']:.1f}% |"
            )
        )


    REPAY = []
    if task_key in ("repayment", "both") and os.path.isdir(_raw):
        for _name in sorted(
            f for f in os.listdir(_raw)
            if f.startswith("repayment_") and f.endswith(".json")
        ):
            _res = json.load(open(os.path.join(_raw, _name)))
            _rows = "\n".join(
                f"| {100 * _r['ratio']:.0f}% | {_r['pre_ppl']:.4f} | "
                f"{_r['post_ppl']:.4f} | {_r['post_delta_pct']:+.2f}% | "
                f"{_r['repair_pct']:.1f}% | {_r['ft_seconds']:.1f} |"
                for _r in _res["rows"]
            )
            REPAY.append(
                mo.md(
                    f"### ♻️ Repayment — {_res['model']}\n\n"
                    f"unpruned PPL (base): {_res['unpruned_ppl']:.4f}\n\n"
                    "| ratio | pre-repayment PPL | post-repayment PPL | ΔPPL after "
                    "| repair % | FT seconds |\n|---|---|---|---|---|---|\n" + _rows
                )
            )


    CARDS = []
    if os.path.isdir(_raw):
        _wanted = tuple(
            f"gpu_{_st}_{_k}_{_tag(ratio_val)}.json"
            for _k in key_list
            for _st in ("summary", "ppl", "latency", "mask", "kv")
        )
        for _name in sorted(os.listdir(_raw)):
            if _name in _wanted or _name.startswith("repayment_"):
                with open(os.path.join(_raw, _name), "rb") as _fh:
                    CARDS.append(mo.download(data=_fh.read(), filename=_name))


    FIGS = []
    if os.path.isdir(_fig_dir):
        for _k in key_list:
            for _fig in ("latency", "kv"):
                _p = os.path.join(_fig_dir, f"gpu_{_fig}_{_k}.png")
                if os.path.exists(_p):
                    FIGS.append(mo.image(src=_p, width=720))

    RESULT = mo.vstack(
        [mo.md(
            f"## ✅ ALL-IN-ONE শেষ ({mins:.1f} মিনিট) — "
            f"task={task_key} · model={model_key}"
        )]
        + (TABLES if TABLES else [mo.md("_বেঞ্চমার্ক টেবিল নেই_")])
        + REPAY
        + ([mo.md("### 🖼 figures")] + FIGS if FIGS else [])
        + ([mo.md("### 📥 download artifacts")] + CARDS
           if CARDS else [mo.md("### 📥 download artifacts"), mo.md("_কিছু নেই_")])
        + [mo.md(
            "**[repo](https://github.com/MD-Mushfiqur123/slm-head-redundancy-benchmark)"
            f" · models: {', '.join(AI_MODELS[:-1])} · "
            f"{len(key_list)} মডেল এই রানে**"
        )]
        + [mo.md("### 📜 run log"),
           mo.accordion(
               {f"📜 full log — {_t}": _log for _t, _log in _tails.items()}
           )]
        + [AI_UI],
    )
    return (RESULT,)


@app.cell
def _(RESULT):
    RESULT
    return


if __name__ == "__main__":
    app.run()
