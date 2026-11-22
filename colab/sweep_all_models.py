# SLM-HEAD-REDUNDANCY-MONUMENT — BUTTON-2 ALL-MODELS GPU SWEEP (standalone)
#
# ▶️ BUTTON A — qwen2_05b RECOVER: crashed run-এর disk-এ থাকা sweep JSON থেকে
#    summary + download cards — GPU ছাড়া, সেকেন্ডে।
# ▶️ BUTTON B — সব ৬ মডেল sweep ধারাবাহিক (ratios 10/20/40/50/60%) — ~2 ঘণ্টা।
# ▶️ BUTTON C — dropdown থেকে একক মডেল sweep (~15-25 min)।
#
# canonical 30%/29.2% measured আর্টিফ্যাক্ট কখনো overwrite হয় না — download cards
# থেকেও বাদ। /TRUTH: প্রতিটা সংখ্যা runtime-এ মাপা — no mocks, no placeholders.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import glob
    import json
    import os
    import subprocess
    import sys
    import time

    import marimo as mo
    return glob, json, mo, os, subprocess, sys, time


@app.cell
def _(mo):
    mo.md(
        """
        # 🎯 BUTTON-2 — ALL-MODELS FULL-RATIO GPU SWEEP

        **বোতাম A (আগে চাপো — সেকেন্ড, GPU লাগে না):** qwen2_05b recover —
        আগের crashed রানের JSON থেকে summary + download cards।
        **বোতাম B (ভারী, ~2 ঘণ্টা):** বাকি ৬ মডেল sweep এক ক্লিকে।
        **বোতাম C:** একক মডেল sweep।

        /TRUTH LAW: প্রতিটা সংখ্যা runtime-এ মাপা — canonical measured আর্টিফ্যাক্ট overwrite হয় না।
        """
    )
    return


@app.cell
def _(mo):
    RECOVER_RUN = mo.ui.run_button(
        label="♻️ BUTTON A — qwen2_05b RECOVER (GPU ছাড়া, সেকেন্ড)"
    )
    ALL_RUN = mo.ui.run_button(
        label="▶️▶️ BUTTON B — সব ৬ মডেল sweep, ratios 10/20/40/50/60% (auto ~2 ঘণ্টা)"
    )
    SWEEP_DROP = mo.ui.dropdown(
        options=["gpt2_small", "gpt2_medium", "smollm_135m", "opt_125m",
                 "pythia_160m", "pythia_410m", "qwen2_05b"],
        value="gpt2_small", label="একক মডেল sweep (BUTTON C-এর জন্য)",
    )
    ONE_RUN = mo.ui.run_button(
        label="▶️ BUTTON C — নির্বাচিত মডেল sweep (~15-25 min)"
    )
    UI_ALL = mo.vstack([
        mo.md("### 🎯 BUTTON A — আগে এটা চাপো (recover, GPU ছাড়া)"),
        RECOVER_RUN,
        mo.md("### 🎯 BUTTON B — সব ৬ মডেল sweep (এক ক্লিকে, দীর্ঘ রান)"),
        ALL_RUN,
        mo.md("### 🎯 BUTTON C — বা একক মডেল"),
        SWEEP_DROP, ONE_RUN,
    ])
    return ALL_RUN, ONE_RUN, RECOVER_RUN, SWEEP_DROP, UI_ALL


@app.cell
def _(mo):
    mo.md("---")
    return


@app.cell
def _(RESULT_RECOVER):
    RESULT_RECOVER
    return


@app.cell
def _(RESULT_SWEEP):
    RESULT_SWEEP
    return


@app.cell
def _(glob, json, mo, os, RECOVER_RUN, UI_ALL):
    _RAW = "slm-head-redundancy-benchmark/results/raw"
    RESULT_RECOVER = mo.md("⏸️ BUTTON A চালু হয়নি।")
    mo.stop(
        not RECOVER_RUN.value,
        mo.vstack([
            mo.md("⏸️ BUTTON A চালু হয়নি — crash হওয়া qwen2_05b sweep recover করতে "
                  "বোতাম A চাপো (এই session-এর disk-এ JSON থাকলে GPU ছাড়াই হবে)।"),
            UI_ALL,
        ]),
    )

    _files = sorted(glob.glob(os.path.join(_RAW, "gpu_summary_qwen2_05b_*.json")))
    _canonical = ("29.2", "30")
    _new = [f for f in _files
            if not any(c in os.path.basename(f) for c in _canonical)]
    mo.stop(
        not _new,
        mo.vstack([
            mo.md("⚠️ এই session-এর disk-এ qwen2_05b-এর নতুন sweep JSON নেই "
                  "(ফ্রেশ session) — তাহলে qwen2_05b-কে BUTTON C দিয়ে আবার sweep করো "
                  "বা BUTTON B-তে যোগ হওয়া প্রতীক্ষা করো।"),
            UI_ALL,
        ]),
    )

    _rows = []
    for _f in sorted(_new, key=lambda p: json.load(open(p))["ratio"]):
        _s = json.load(open(_f))
        _tag = ("%.1f" % (100 * _s["ratio"])).replace(".0", "")
        _r = _s["retention"]
        _kv = list(_s["kv_cache"]["seq_lens"].values())[-1]["reduction_pct"]
        _rows.append(
            f"| {_tag}% | {_r['wt2_unpruned']['ppl']:.4f} | "
            f"{_r['wt2_pruned']['ppl']:.4f} | "
            f"{100 * _r['lambada_unpruned']['top1_accuracy']:.1f}% | "
            f"{100 * _r['lambada_pruned']['top1_accuracy']:.1f}% | {_kv:.2f}% |"
        )
    _cards = []
    for _p in sorted(glob.glob(os.path.join(_RAW, "gpu_*_qwen2_05b_*.json"))):
        _nm = os.path.basename(_p)
        if "29.2" in _nm or "30" in _nm:
            continue
        with open(_p, "rb") as _fd:
            _cards.append(mo.download(data=fh.read(), filename=_nm))
    if os.path.isdir("slm-head-redundancy-benchmark/results/figures"):
        for _fig in ("latency", "kv"):
            _p = os.path.join("slm-head-redundancy-benchmark/results/figures",
                              f"gpu_{_fig}_qwen2_05b.png")
            if os.path.exists(_p):
                with open(_p, "rb") as _fd:
                    _cards.append(mo.download(data=fh.read(), filename=os.path.basename(_p)))

    RESULT_RECOVER = mo.vstack([
        mo.md(f"## ♻️ qwen2_05b RECOVER শেষ — {len(_new)}টি সুইপ ট্যাগ পাওয়া গেছে "
              f"(ক্যানোনিক্যাল 29.2% বাদ)"),
        mo.md(
            "### 📊 qwen2_05b — full-ratio GPU matrix (RTX Pro 6000, fp16)\n\n"
            "| excision | WT2 base | WT2 pruned | LAMB base | LAMB pruned | KV cut |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(_rows)
        ),
        mo.md("### 📥 download sweep artifacts (canonical 29.2% বাদ)") + mo.hstack(_cards)
        if _cards else mo.md("আর্টিফ্যাক্ট নেই"),
        UI_ALL,
    ])
    return (RESULT_RECOVER,)


@app.cell
def _(glob, json, mo, os, subprocess, sys, time, UI_ALL, ALL_RUN, ONE_RUN,
      SWEEP_DROP):
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
            return "CUDA_UNKNOWN", _r.stderr[-400:] or "torch import failed"
        _lines = _r.stdout.strip().splitlines()
        return _lines[0], (_lines[1] if len(_lines) > 1 else "unknown")


    _gpu_flag, _gpu_name = _probe_gpu()
    if _gpu_flag != "CUDA_OK":
        mo.stop(
            True,
            mo.vstack([
                mo.md("## ⛔ GPU পাওয়া যায়নি — molab-এ RTX Pro 6000 সিলেক্ট করে রিস্টার্ট করো"),
                UI_ALL,
            ]),
        )
    mo.stop(
        not (ALL_RUN.value or ONE_RUN.value),
        mo.vstack([
            mo.md(f"⏸️ কোনো sweep বোতাম চালু হয়নি — GPU রেডি ✅ ({_gpu_name})। "
                  "নিচে BUTTON B (সব ৬ মডেল) বা BUTTON C (একক) চাপো।"),
            UI_ALL,
        ]),
    )
    _ALL6 = ["gpt2_small", "gpt2_medium", "smollm_135m", "opt_125m",
             "pythia_160m", "pythia_410m"]
    _MODELS = _ALL6 if ALL_RUN.value else [SWEEP_DROP.value or "gpt2_small"]
    _RATIOS = ["0.1", "0.2", "0.4", "0.5", "0.6"]
    _banner(f"① GPU preflight ok — {_gpu_name} · sweep models: {', '.join(_MODELS)}")

    if not os.path.isdir(_REPO):
        _banner("② cloning repo")
        subprocess.run(["git", "clone", "--depth", "1", _REPO_URL, _REPO], check=True)
    else:
        _banner("② pulling latest repo")
        subprocess.run(
            ["git", "-C", _REPO, "fetch", "origin", "main"],
            capture_output=True, text=True, check=True,
        )
        if subprocess.run(
            ["git", "-C", _REPO, "merge", "--ff-only", "origin/main"],
            capture_output=True, text=True,
        ).returncode != 0:
            subprocess.run(
                ["git", "-C", _REPO, "reset", "--hard", "origin/main"],
                capture_output=True, text=True, check=True,
            )
    for _pkg in ("transformers", "datasets", "matplotlib", "tqdm"):
        try:
            __import__(_pkg)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)

    _repo_abs = os.path.abspath(_REPO)
    _raw_abs = os.path.abspath(os.path.join(_REPO, "results", "raw"))
    _fig_abs = os.path.abspath(os.path.join(_REPO, "results", "figures"))

    _tails = {}
    _fails = []
    for _sel in _MODELS:
        for _r in _RATIOS:
            _ts = time.perf_counter()
            _banner(f"▶ sweep·{_sel}·ratio {_r}")
            _p = subprocess.run(
                [sys.executable, "scripts/gpu_extension.py",
                 "--model", _sel, "--ratio", _r, "--fp16"],
                capture_output=True, text=True, cwd=_repo_abs,
            )
            _mins = (time.perf_counter() - _ts) / 60
            _tails[f"gpu·{_sel}·r={_r}"] = (f"[{_mins:.1f} min, exit {_p.returncode}]\n"
                                            + _p.stdout[-12000:] + "\n"
                                            + _p.stderr[-4000:])
            if _p.returncode != 0:
                _fails.append((f"gpu·{_sel}·r={_r}", _p.returncode, _p.stderr[-1500:]))

    _model_tables = []
    for _sel in _MODELS:
        _rows = []
        for _f in sorted(glob.glob(os.path.join(_raw_abs, f"gpu_summary_{_sel}_*.json"))):
            _bn = os.path.basename(_f)
            if "30" in _bn or "29.2" in _bn:
                continue
            _s = json.load(open(_f))
            _kv = list(_s["kv_cache"]["seq_lens"].values())[-1]["reduction_pct"]
            _r = _s["retention"]
            _rows.append(
                f"| {_s['ratio']:.3f} | {_r['wt2_unpruned']['ppl']:.4f} | "
                f"{_r['wt2_pruned']['ppl']:.4f} | "
                f"{100 * _r['lambada_unpruned']['top1_accuracy']:.1f}% | "
                f"{100 * _r['lambada_pruned']['top1_accuracy']:.1f}% | {_kv:.2f}% |"
            )
        if _rows:
            _model_tables.append(mo.md(
                f"### 📊 {_sel} — sweep matrix (fp16)\n\n"
                "| excision | WT2 base | WT2 pruned | LAMB base | LAMB pruned | KV cut |\n"
                "|---|---|---|---|---|---|\n" + "\n".join(_rows)
            ))

    _cards = []
    for _sel in _MODELS:
        for _p in sorted(glob.glob(os.path.join(_raw_abs, f"gpu_*_{_sel}_*.json"))):
            _nm = os.path.basename(_p)
            if "30" in _nm or "29.2" in _nm:
                continue
            with open(_p, "rb") as _fd:
                _cards.append(mo.download(data=fh.read(), filename=_nm))
        for _fig in ("latency", "kv"):
            _p = os.path.join(_fig_abs, f"gpu_{_fig}_{_sel}.png")
            if os.path.exists(_p):
                with open(_p, "rb") as _fd:
                    _cards.append(mo.download(data=fh.read(), filename=os.path.basename(_p)))

    _mins_total = (time.perf_counter() - _t0) / 60
    _fail_md = (
        [mo.md("### ⚠️ ফেল হওয়া রান — রিপোর্ট এজেন্টকে দাও")]
        + [mo.accordion({f"⚠️ {_fn} (exit {_rc})": _err for _fn, _rc, _err in _fails})]
        if _fails else []
    )

    RESULT_SWEEP = mo.vstack(
        [mo.md(
            f"## ✅ ALL-MODELS SWEEP শেষ ({_mins_total:.1f} মিনিট) — "
            f"models: {', '.join(_MODELS)} · ratios 10/20/40/50/60%"
            + ("" if not _fails else " · ⚠️ কিছু রান ফেল")
        )]
        + ([mo.md("## 📥 এখন এগুলো এজেন্টকে paste/upload করো (সব মডেলের)")] if _MODELS else [])
        + (_model_tables or [mo.md("⚠️ কোনো sweep summary তৈরি হয়নি")])
        + ([mo.md("### 📥 download sweep artifacts (canonical 30%/29.2% বাদ)")] + _cards
           if _cards else [mo.md("_নতুন sweep আর্টিফ্যাক্ট নেই_")])
        + _fail_md
        + [mo.md("### 📜 run logs"),
           mo.accordion({f"📜 {_t}": _log for _t, _log in _tails.items()})
           if _tails else mo.md("কোনো রান হয়নি")]
        + [UI_ALL],
    )
    return (RESULT_SWEEP,)


if __name__ == "__main__":
    app.run()
