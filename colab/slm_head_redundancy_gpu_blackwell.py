# SLM-HEAD-REDUNDANCY-MONUMENT — GPU ALL-IN-ONE (Blackwell v2)
#
# একটাই অপারেশন সেল, একটাই বোতাম:
#   ▶️ preflight GPU → pull → install → run → tables → figures → download
#
# task = benchmark → scripts/gpu_extension.py (সেভেন-মডেল সাপোর্ট, GQA auto-snap)
# task = repayment → scripts/finetune_repayment.py (fine-tuning repayment sweep)
# task = both      → দুটোই পরপর
#
# v2 hardening (মলব ক্যাশ-বাস্ট নাম):
#   • আর্টিফ্যাক্ট টেবিল/ডাউনলোড glob-বেসড — GQA snap-এ "29.2" ট্যাগও ধরে (qwen2_05b)
#   • এক রান ফেল করলেও বাকিগুলো শেষ হয়; শেষে ⚠️ ফেল-রিপোর্ট
#   • GPU preflight — CUDA না থাকলে ১০ মিনিটের রানের আগেই বলে দেয়
#   • fps পছন্দ (fp16 on/off) অনুযায়ী সঠিক summary ফাইল বাছে
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
        # 🗽 SLM-HEAD-REDUNDANCY-MONUMENT — GPU ALL-IN-ONE v2

        **একটাই অপারেশন সেল, একটাই বোতাম।** বোতাম চাপলে: ① repo pull ② dependency
        install ③ benchmark/repayment ④ measured টেবিল ⑤ প্লট ⑥ সব JSON/CSV ডাউনলোড।

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
        value="both",
        label="task",
    )
    AI_MODEL = mo.ui.dropdown(
        options=AI_MODELS,
        value="all",
        label="model (all = পুর্ণ ৭-মডেল ম্যাট্রিক্স)",
    )
    AI_RATIO = mo.ui.text(value="0.30", label="pruning ratio")
    AI_FP16 = mo.ui.switch(label="fp16 latency/KV profile", value=True)
    AI_FORCE = mo.ui.switch(label="force rerun (ক্যাশ বাদ দিয়ে নতুন করে)", value=False)
    AI_RUN = mo.ui.run_button(
        label="▶️ ALL-IN-ONE: preflight → pull → run → tables → download"
    )
    AI_UI = mo.vstack([
        mo.md(
            "### 🎯 ALL-IN-ONE\n\n"
            "setting বদলে বোতামে চাপ দাও — বাকি সব এক সেলেই।"
        ),
        AI_TASK, AI_MODEL, AI_RATIO, AI_FP16, AI_FORCE, AI_RUN,
    ])
    return (
        AI_FP16, AI_FORCE, AI_MODELS, AI_MODEL, AI_RATIO, AI_RUN,
        AI_TASK, AI_UI,
    )


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
    _t0 = time.perf_counter()


    def _banner(text: str) -> None:
        print("─" * 64, flush=True)
        print(text, flush=True)
        print("─" * 64, flush=True)


    def _probe_gpu():
        _r = subprocess.run(
            [sys.executable, "-c",
             "import torch;"
             "print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NO');"
             "print(torch.cuda.get_device_name(0)"
             "      if torch.cuda.is_available() else 'cpu')"],
            capture_output=True, text=True,
        )
        if _r.returncode != 0:
            return "CUDA_UNKNOWN", _r.stderr[-400:] or "torch import failed (no stderr)"
        _lines = _r.stdout.strip().splitlines()
        return _lines[0], (_lines[1] if len(_lines) > 1 else "unknown")


    _gpu_flag, _gpu_name = _probe_gpu()
    if _gpu_flag != "CUDA_OK":
        mo.stop(
            True,
            mo.vstack([
                mo.md(
                    "## ⛔ GPU পাওয়া যায়নি\n\n"
                    f"preflight: `{_gpu_flag}` — runtime-এ CUDA exclusive দরকার।\n\n"
                    "molab-এ: নোটবুক স্পেক-gear ⚙️ → GPU **NVIDIA RTX Pro 6000** সিলেক্ট করে "
                    "সেশন রিস্টার্ট করো, তারপর আবার বোতাম চাপো।"
                ),
                AI_UI,
            ]),
        )
    else:
        mo.stop(
            not AI_RUN.value,
            mo.vstack([
                mo.md(
                    f"⏸️ ALL-IN-ONE এখনো চালু হয়নি — GPU রেডি ✅ ({_gpu_name})।\n\n"
                    "উপরের বোতামে চাপ দাও।"
                ),
                AI_UI,
            ]),
        )
    _banner(f"② GPU preflight ok — {_gpu_name}")


    if not os.path.isdir(_REPO):
        _banner("③ cloning repo")
        subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, _REPO], check=True
        )
    else:
        _banner("③ pulling latest repo")
        subprocess.run(
            ["git", "-C", _REPO, "fetch", "origin", "main"],
            capture_output=True, text=True, check=True,
        )
        if subprocess.run(
            ["git", "-C", _REPO, "merge", "--ff-only", "origin/main"],
            capture_output=True, text=True,
        ).returncode != 0:
            # Untracked artifacts from earlier runs in this session can block a
            # fast-forward when a pushed update touches the same paths. Reset:
            # committed measured data is identical, locally-only artifacts stay.
            _banner("④ fast-forward blocked by local artifacts — resetting to origin/main")
            subprocess.run(
                ["git", "-C", _REPO, "reset", "--hard", "origin/main"],
                capture_output=True, text=True, check=True,
            )

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
    _model_keys = AI_MODELS[:-1] if model_key == "all" else [model_key]
    for _mk in _model_keys:
        if _mk in _kv_group:
            _g = _kv_group[_mk]
            _n = round(_total_heads[_mk] * ratio_val)
            if _n % _g != 0:
                _snap = (_n // _g) * _g
                _snapped = _snap / _total_heads[_mk]
                _banner(
                    f"note: {_mk} kv_group={_g} — gpu_extension auto-snaps "
                    f"{ratio_val:.4f} -> {_snapped:.6f} ({_snap} heads); ok."
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
        for _rm in _model_keys:
            runs.append(
                (f"repayment·{_rm}", [
                    sys.executable, "scripts/finetune_repayment.py",
                    "--model", _rm,
                ] + (["--force"] if AI_FORCE.value else []))
            )

    _tails = {}
    _fails = []
    for _task_name, _cmd in runs:
        _ts = time.perf_counter()
        _banner(f"④ running: {_task_name} | model={model_key} | ratio={ratio_val}")
        _proc = subprocess.run(
            _cmd, capture_output=True, text=True, cwd=os.path.abspath(_REPO),
        )
        _mins = (time.perf_counter() - _ts) / 60
        _tails[_task_name] = (
            f"[{_mins:.1f} min, exit {_proc.returncode}]\n"
            + _proc.stdout[-20000:] + "\n" + _proc.stderr[-4000:]
        )
        if _proc.returncode != 0:
            _fails.append((_task_name, _proc.returncode, _proc.stderr[-1500:]))
            _banner(f"⚠️ {_task_name} exited {_proc.returncode} — এগিয়ে যাচ্ছি")
        else:
            _banner(f"✓ {_task_name} done ({_mins:.1f} min)")
    mins = (time.perf_counter() - _t0) / 60

    _raw = os.path.join(_REPO, "results", "raw")
    _fig_dir = os.path.join(_REPO, "results", "figures")


    def _pick(k: str, stage: str):
        _cands = sorted(
            os.path.join(_raw, f)
            for f in os.listdir(_raw)
            if f.startswith(f"gpu_{stage}_{k}_") and f.endswith(".json")
        ) if os.path.isdir(_raw) else []
        if not _cands:
            return None
        _pref = [c for c in _cands if ("fp16" in c) == bool(AI_FP16.value)]
        return max(_pref or _cands, key=os.path.getmtime)


    TABLES = []
    for _k in _model_keys:
        _p = _pick(_k, "summary")
        if _p is None:
            TABLES.append(
                mo.md(f"### 📊 {_k} — এই রানে আর্টিফ্যাক্ট পাওয়া যায়নি ⚠️")
            )
            continue
        _s = json.load(open(_p))
        _tag_file = os.path.basename(_p).replace(
            f"gpu_summary_{_k}_", ""
        ).replace(".json", "")
        _speed = _s.get("latency", {}).get("batch_sizes", {})
        _sp1 = _speed.get("1", {}).get("speedup_pct")
        _sp8 = _speed.get("8", {}).get("speedup_pct")
        _sp_txt = (
            "n/a" if _sp1 is None else
            f"{_sp1:+.1f}% / {'n/a' if _sp8 is None else f'{_sp8:+.1f}%'}"
        )
        TABLES.append(
            mo.md(
                f"### 📊 {_k} — measured summary (/TRUTH, runtime)\n\n"
                f"artifact: `{_tag_file}`\n\n"
                "| GPU | heads excised | WT-2 PPL base | WT-2 PPL pruned "
                "| LAMBADA base | LAMBADA pruned | KV cut | speedup b=1/b=8 |\n"
                "|---|---|---|---|---|---|---|---|\n"
                f"| {_s['gpu']} | {_s['excised_heads']}/{_s['total_heads']} | "
                f"{_s['retention']['wt2_unpruned']['ppl']:.4f} | "
                f"{_s['retention']['wt2_pruned']['ppl']:.4f} | "
                f"{100 * _s['retention']['lambada_unpruned']['top1_accuracy']:.1f}% | "
                f"{100 * _s['retention']['lambada_pruned']['top1_accuracy']:.1f}% | "
                f"{list(_s['kv_cache']['seq_lens'].values())[-1]['reduction_pct']:.1f}% | "
                f"{_sp_txt} |"
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
    _seen = set()
    if os.path.isdir(_raw):
        for _name in sorted(os.listdir(_raw)):
            if not _name.endswith(".json"):
                continue
            if not (
                any(_name.startswith(f"gpu_{_st}_{_k}_")
                    for _k in _model_keys
                    for _st in ("summary", "ppl", "latency", "mask", "kv"))
                or _name.startswith("repayment_")
            ):
                continue
            if _name not in _seen:
                _seen.add(_name)
                with open(os.path.join(_raw, _name), "rb") as _fh:
                    CARDS.append(mo.download(data=_fh.read(), filename=_name))
    if os.path.isdir(_REPO):
        for _name in sorted(os.listdir(os.path.join(_REPO, "results"))):
            if _name.startswith("repayment_") and _name.endswith(".csv"):
                with open(os.path.join(_REPO, "results", _name), "rb") as _fh:
                    CARDS.append(mo.download(data=_fh.read(), filename=_name))
    if os.path.isdir(_fig_dir):
        for _k in _model_keys:
            for _fig in ("latency", "kv"):
                _p = os.path.join(_fig_dir, f"gpu_{_fig}_{_k}.png")
                if os.path.exists(_p):
                    with open(_p, "rb") as _fh:
                        CARDS.append(
                            mo.download(data=_fh.read(), filename=f"gpu_{_fig}_{_k}.png")
                        )

    FIGS = []
    if os.path.isdir(_fig_dir):
        for _k in _model_keys:
            for _fig in ("latency", "kv"):
                _p = os.path.join(_fig_dir, f"gpu_{_fig}_{_k}.png")
                if os.path.exists(_p):
                    FIGS.append(mo.image(src=_p, width=720))

    _fail_md = (
        [mo.md("### ⚠️ ফেল হওয়া রান — বাকিগুলো তবু শেষ হয়েছে")]
        + [mo.accordion({f"⚠️ {_fn} (exit {_rc})": _err for _fn, _rc, _err in _fails})]
        if _fails else []
    )

    RESULT = mo.vstack(
        [mo.md(
            f"## ✅ ALL-IN-ONE শেষ ({mins:.1f} মিনিট) — "
            f"task={task_key} · model={model_key} · GPU {_gpu_name}"
            + ("" if not _fails else " · ⚠️ কিছু রান ফেল (নিচে রিপোর্ট)")
        )]
        + (TABLES if TABLES else [mo.md("_বেঞ্চমার্ক টেবিল নেই_")])
        + REPAY
        + ([mo.md("### 🖼 figures")] + FIGS if FIGS else [])
        + ([mo.md("### 📥 download artifacts (JSON/CSV/PNG)")]
           + CARDS if CARDS else
           [mo.md("### 📥 download artifacts"), mo.md("_কিছু নেই_")])
        + _fail_md
        + [mo.md(
            "**[repo](https://github.com/MD-Mushfiqur123/slm-head-redundancy-benchmark)"
            f" · models: {', '.join(AI_MODELS[:-1])} · "
            f"{len(_model_keys)} মডেল এই রানে · ratio={ratio_val} · fp16={AI_FP16.value}"
            " · /TRUTH: প্রতিটা সংখ্যা runtime-এ মাপা"
        )]
        + [mo.md("### 📜 run log"),
           mo.accordion(
               {f"📜 full log — {_t}": _log for _t, _log in _tails.items()}
           )
           if _tails else mo.md("_কোনো রান হয়নি_")]
        + [AI_UI],
    )
    return (RESULT,)


@app.cell
def _(RESULT):
    RESULT
    return


if __name__ == "__main__":
    app.run()
