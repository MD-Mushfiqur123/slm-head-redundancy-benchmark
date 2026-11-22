# SLM-HEAD-REDUNDANCY-MONUMENT — FINISH ALL GAPS (standalone)
#
# গবেষণা সম্পূর্ণ শেষ করার ফাইনাল নোটবুক। দুটো আলাদা বোতাম:
#   ▶️ BUTTON 1 — LIGHT GAPS (auto, ~12-15 মিনিট):
#        A. opt_125m repayment @ 10/20/40%  → canonical JSON-এ merge (0.3 untouched)
#        B. budget robustness: gpt2_small + qwen2_05b @ r=0.3, epochs 1 এবং 4
#        C. seed robustness: gpt2_small @ r=0.3, seeds 8/9
#   ▶️ BUTTON 2 — FULL-RATIO GPU SWEEP (প্রতি মডেল ~15-25 মিনিট):
#        dropdown থেকে মডেল সিলেক্ট → ratios 10/20/40/50/60% সব stage
#
# canonical measured আর্টিফ্যাক্ট কখনো overwrite হয় না — নতুন ফাইল নতুন নামে
# (suffix _fill/_e1/_e4/_seed8/_seed9), merge শুধু নতুন ratio row যোগ করে।
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
        # 🏁 FINISH ALL GAPS — গবেষণা সম্পূর্ণকরণ

        **বোতাম 1 (আগে চাপো):** opt repayment 10/20/40% merge + budget (1/4-epoch) +
        seed (8/9) robustness — সব অটো, ~12-15 মিনিট।
        **বোতাম 2 (ঐচ্ছিক, প্রতি মডেল):** full-ratio GPU sweep 10→60%।

        /TRUTH LAW: প্রতিটা সংখ্যা runtime-এ মাপা — canonical measured আর্টিফ্যাক্ট
        overwrite হয় না।
        """
    )
    return


@app.cell
def _(mo):
    LIGHT_RUN = mo.ui.run_button(
        label="▶️ BUTTON 1 — LIGHT GAPS: opt fill + budget + seed robustness (auto ~12-15 min)"
    )
    SWEEP_DROP = mo.ui.dropdown(
        options=["gpt2_small", "gpt2_medium", "opt_125m", "pythia_160m",
                 "pythia_410m", "smollm_135m", "qwen2_05b"],
        value="gpt2_small", label="sweep model (BUTTON 2-এর জন্য)",
    )
    SWEEP_RUN = mo.ui.run_button(
        label="▶️ BUTTON 2 — FULL-RATIO SWEEP (10/20/40/50/60%) — নির্বাচিত মডেল"
    )
    UI_ALL = mo.vstack([
        mo.md("### 🎯 BUTTON 1 — আগে এটা চাপো (এক ক্লিকে সব light gaps)"),
        LIGHT_RUN,
        mo.md("### 🎯 BUTTON 2 — ঐচ্ছিক, হেভি: মডেল সিলেক্ট করে সুইপ"),
        SWEEP_DROP, SWEEP_RUN,
    ])
    return LIGHT_RUN, SWEEP_DROP, SWEEP_RUN, UI_ALL


@app.cell
def _(mo):
    mo.md("---")
    return


@app.cell
def _(glob, json, LIGHT_RUN, mo, os, subprocess, sys, time, UI_ALL):
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
                mo.md(
                    "## ⛔ GPU পাওয়া যায়নি\n\n"
                    f"preflight: `{_gpu_flag}` — এই নোটবুক CUDA exclusive।\n\n"
                    "molab-এ: স্পেক-গিয়ার ⚙️ → GPU **NVIDIA RTX Pro 6000** সিলেক্ট করে "
                    "সেশন রিস্টার্ট করো, তারপর আবার নিচের **BUTTON 1** চাপো।"
                ),
                UI_ALL,
            ]),
        )
    mo.stop(
        not LIGHT_RUN.value,
        mo.vstack([
            mo.md(
                f"⏸️ এখনো চালু হয়নি — GPU রেডি ✅ ({_gpu_name})।\n\n"
                "নিচের **BUTTON 1** চাপো।"
            ),
            UI_ALL,
        ]),
    )
    _banner(f"① GPU preflight ok — {_gpu_name}")


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
            _banner("② fast-forward blocked — resetting to origin/main")
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
    _res_abs = os.path.abspath(os.path.join(_REPO, "results"))

    _tails = {}
    _fails = []


    def _run(desc, argv):
        _ts = time.perf_counter()
        _banner(f"▶ {desc}")
        _p = subprocess.run(
            [sys.executable] + argv, capture_output=True, text=True, cwd=_repo_abs,
        )
        _mins = (time.perf_counter() - _ts) / 60
        _tails[desc] = (f"[{_mins:.1f} min, exit {_p.returncode}]\n"
                        + _p.stdout[-16000:] + "\n" + _p.stderr[-4000:])
        if _p.returncode != 0:
            _fails.append((desc, _p.returncode, _p.stderr[-1500:]))
            _banner(f"⚠️ {desc} exit {_p.returncode} — পরের পর্বে যাচ্ছি")
        else:
            _banner(f"✓ {desc} done ({_mins:.1f} min)")
        return _p.returncode == 0


    _CSV_HEAD = ["ratio", "pre_ppl", "post_ppl", "pre_delta_pct",
                 "post_delta_pct", "repair_pct", "ft_seconds"]


    def _write_csv(path, rows):
        import csv as _csv
        with open(path, "w", newline="") as _f:
            _w = _csv.writer(_f)
            _w.writerow(_CSV_HEAD)
            for _r in rows:
                _w.writerow([_r["ratio"], _r["pre_ppl"], _r["post_ppl"],
                             _r["pre_delta_pct"], _r["post_delta_pct"],
                             _r["repair_pct"], _r["ft_seconds"]])


    # ── ③ A: opt_125m repayment 10/20/40% + canonical merge ──────────────
    _opt_canon = os.path.join(_raw_abs, "repayment_opt_125m.json")
    _opt_csv = os.path.join(_res_abs, "repayment_opt_125m.csv")
    _fill_json = os.path.join(_raw_abs, "repayment_opt_125m_fill.json")
    _fill_csv = os.path.join(_res_abs, "repayment_opt_125m_fill.csv")
    _opt_merged = False
    if os.path.exists(_opt_json := _opt_canon):
        _old = json.load(open(_opt_json))
        _old_rows = list(_old["rows"])
        _banner(f"③ A: canonical repayment_opt_125m.json আছে ({len(_old_rows)} row) — protected")
    else:
        _old = None
        _old_rows = []
        _banner("③ A: canonical opt repayment JSON নেই — নতুন run canon হবে")
    if _run("repayment·opt_125m·ratios 0.1 0.2 0.4 · suffix _fill",
            ["scripts/finetune_repayment.py", "--model", "opt_125m",
             "--ratios", "0.1", "0.2", "0.4", "--out-suffix", "_fill"]):
        _fill = json.load(open(_fill_json))
        _have = {round(_r["ratio"], 6) for _r in _old_rows}
        _fresh = [dict(_r) for _r in _fill["rows"]
                  if round(_r["ratio"], 6) not in _have]
        _all_rows = sorted(_old_rows + _fresh, key=lambda _r: _r["ratio"])
        _top = dict(_old) if _old else dict(_fill)
        _top["rows"] = _all_rows
        json.dump(_top, open(_opt_json, "w"), indent=2)
        _write_csv(_opt_csv, _all_rows)
        os.remove(_fill_json)
        if os.path.exists(_fill_csv):
            os.remove(_fill_csv)
        _opt_merged = True
        _banner(f"③ A: merge done — canonical opt repayment এখন {len(_all_rows)} row "
                "(0.3 row untouched)")

    # ── ③ B: budget robustness (epochs 1 vs 2-canonical vs 4) ────────────
    _budget_specs = [
        ("gpt2_small", 1, "_e1"), ("gpt2_small", 4, "_e4"),
        ("qwen2_05b", 1, "_e1"), ("qwen2_05b", 4, "_e4"),
    ]
    _budget_ok = {}
    for _model, _ep, _sfx in _budget_specs:
        _ok = _run(
            f"repayment·{_model}·epochs {_ep}·suffix {_sfx}",
            ["scripts/finetune_repayment.py", "--model", _model,
             "--ratios", "0.3", "--epochs", str(_ep), "--out-suffix", _sfx],
        )
        _budget_ok[(_model, _sfx)] = _ok

    # ── ③ C: seed robustness (gpt2_small r=0.3, seeds 8/9; seed 7 = canonical) ──
    _seed_ok = {}
    for _seed in (8, 9):
        _sfx = f"_seed{_seed}"
        _ok = _run(
            f"repayment·gpt2_small·seed {_seed}·suffix {_sfx}",
            ["scripts/finetune_repayment.py", "--model", "gpt2_small",
             "--ratios", "0.3", "--seed", str(_seed), "--out-suffix", _sfx],
        )
        _seed_ok[_sfx] = _ok

    mins = (time.perf_counter() - _t0) / 60

    # ── ④ tables ─────────────────────────────────────────────────────────
    TABLES = []

    if os.path.exists(_opt_json):
        _res = json.load(open(_opt_json))
        _rows = "\n".join(
            f"| {100 * _r['ratio']:.0f}% | {_r['pre_ppl']:.4f} | "
            f"{_r['post_ppl']:.4f} | {_r['post_delta_pct']:+.2f}% | "
            f"{_r['repair_pct']:.1f}% | {_r['ft_seconds']:.1f} |"
            for _r in _res["rows"]
        )
        TABLES.append(mo.md(
            "### ♻️ Repayment — opt_125m (মার্জড canonical, সব রেশিও)\n\n"
            f"unpruned PPL (base): {_res['unpruned_ppl']:.4f}\n\n"
            "| ratio | pre PPL | post PPL | ΔPPL after | repair % | FT s |\n"
            "|---|---|---|---|---|---|\n" + _rows
        ))

    _brows = []
    for _model in ("gpt2_small", "qwen2_05b"):
        for _sfx, _lab in (("_e1", "1 epoch"), ("", "2 epoch (canonical)"), ("_e4", "4 epochs")):
            _p = os.path.join(_raw_abs, f"repayment_{_model}{_sfx}.json")
            if not os.path.exists(_p):
                continue
            _d = json.load(open(_p))
            _row = min(_d["rows"], key=lambda r: abs(r["ratio"] - 0.3))
            _brows.append(f"| {_model} | {_lab} | {_row['pre_ppl']:.4f} | "
                          f"{_row['post_ppl']:.4f} | {_row['repair_pct']:.1f}% | "
                          f"{_row['ft_seconds']:.1f} |")
    if _brows:
        TABLES.append(mo.md(
            "### 📈 Budget robustness — repair % vs fine-tuning budget (r≈0.3)\n\n"
            "| model | budget | pre PPL | post PPL | repair % | FT s |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(_brows)
        ))

    _srows = []
    for _sfx in ("_seed8", "_seed9"):
        _p = os.path.join(_raw_abs, f"repayment_gpt2_small{_sfx}.json")
        if os.path.exists(_p):
            _d = json.load(open(_p))
            _row = min(_d["rows"], key=lambda r: abs(r["ratio"] - 0.3))
            _srows.append(f"| {_d.get('seed', _sfx.split('seed')[1])} | "
                          f"{_row['pre_ppl']:.4f} | {_row['post_ppl']:.4f} | "
                          f"{_row['repair_pct']:.1f}% |")
    _p7 = os.path.join(_raw_abs, "repayment_gpt2_small.json")
    if os.path.exists(_p7):
        _d7 = json.load(open(_p7))
        _r7 = min(_d7["rows"], key=lambda r: abs(r["ratio"] - 0.3))
        _srows.insert(0, f"| 7 (canonical) | {_r7['pre_ppl']:.4f} | "
                         f"{_r7['post_ppl']:.4f} | {_r7['repair_pct']:.1f}% |")
    if _srows:
        TABLES.append(mo.md(
            "### 🎲 Seed robustness — gpt2_small r=0.3 repair % across seeds\n\n"
            "| seed | pre PPL | post PPL | repair % |\n|---|---|---|---|\n"
            + "\n".join(_srows)
        ))

    # ── ⑤ downloads ──────────────────────────────────────────────────────
    CARDS = []
    _want = [
        "repayment_opt_125m.json", "repayment_gpt2_small_e1.json",
        "repayment_gpt2_small_e4.json", "repayment_qwen2_05b_e1.json",
        "repayment_qwen2_05b_e4.json", "repayment_gpt2_small_seed8.json",
        "repayment_gpt2_small_seed9.json",
    ]
    for _name in _want:
        _p = os.path.join(_raw_abs, _name)
        if os.path.exists(_p):
            with open(_p, "rb") as _fh:
                CARDS.append(mo.download(data=_fh.read(), filename=_name))
    if os.path.exists(_opt_csv):
        with open(_opt_csv, "rb") as _fh:
            CARDS.append(mo.download(data=_fh.read(), filename="repayment_opt_125m.csv"))

    _fail_md = (
        [mo.md("### ⚠️ ফেল হওয়া রান — নিচের রিপোর্টটা এজেন্টকে দাও")]
        + [mo.accordion({f"⚠️ {_fn} (exit {_rc})": _err for _fn, _rc, _err in _fails})]
        if _fails else []
    )

    RESULT_LIGHT = mo.vstack(
        [mo.md(
            f"## ✅ FINISH · LIGHT GAPS শেষ ({mins:.1f} মিনিট) — GPU {_gpu_name}"
            + (" · opt merge ✓" if _opt_merged else "")
            + ("" if not _fails else " · ⚠️ কিছু রান ফেল")
        )]
        + TABLES
        + ([mo.md("### 📥 download artifacts (নতুন measured JSON/CSV)")] + CARDS
           if CARDS else [mo.md("_কিছু নতুন আর্টিফ্যাক্ট নেই_")])
        + _fail_md
        + [mo.md("**[repo](https://github.com/MD-Mushfiqur123/slm-head-redundancy-benchmark)"
                 " · /TRUTH: প্রতিটা সংখ্যা runtime-এ মাপা · canonical overwrite নেই")]
        + [mo.md("### 📜 run logs"),
           mo.accordion({f"📜 {_t}": _log for _t, _log in _tails.items()})
           if _tails else mo.md("_কোনো রান হয়নি_")]
        + [UI_ALL],
    )
    return (RESULT_LIGHT,)


@app.cell
def _(RESULT_LIGHT):
    RESULT_LIGHT
    return


@app.cell
def _(glob, json, mo, os, subprocess, SWEEP_DROP, SWEEP_RUN, sys, time, UI_ALL):
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
                mo.md(
                    "## ⛔ GPU পাওয়া যায়নি — molab-এ RTX Pro 6000 সিলেক্ট করে রিস্টার্ট করো"),
                UI_ALL,
            ]),
        )
    mo.stop(
        not SWEEP_RUN.value,
        mo.vstack([
            mo.md(
                f"⏸️ BUTTON 2 চালু হয়নি — GPU রেডি ✅ ({_gpu_name})। "
                "নিচে মডেল সিলেক্ট করে বোতাম চাপো (~15-25 min/model)।"),
            UI_ALL,
        ]),
    )
    _SEL = SWEEP_DROP.value or "gpt2_small"
    _banner(f"① GPU preflight ok — {_gpu_name} · sweep model: {_SEL}")

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
    _RATIOS = ["0.1", "0.2", "0.4", "0.5", "0.6"]
    for _r in _RATIOS:
        _ts = time.perf_counter()
        _banner(f"▶ sweep·{_SEL}·ratio {_r}")
        _p = subprocess.run(
            [sys.executable, "scripts/gpu_extension.py",
             "--model", _SEL, "--ratio", _r, "--fp16"],
            capture_output=True, text=True, cwd=_repo_abs,
        )
        _mins = (time.perf_counter() - _ts) / 60
        _tails[f"gpu·{_SEL}·r={_r}"] = (f"[{_mins:.1f} min, exit {_p.returncode}]\n"
                                        + _p.stdout[-16000:] + "\n" + _p.stderr[-4000:])
        if _p.returncode != 0:
            _fails.append((f"gpu·{_SEL}·r={_r}", _p.returncode, _p.stderr[-1500:]))

    # which summary tags exist now for this model
    _tags = []
    if os.path.isdir(_raw_abs):
        for _f in sorted(glob.glob(os.path.join(_raw_abs, f"gpu_summary_{_SEL}_*.json"))):
            _mid = os.path.basename(_f)[len(f"gpu_summary_{_SEL}_"):-len(".json")]
            if _mid.endswith("_fp16"):
                _mid = _mid[:-len("_fp16")]
            _tags.append(_mid)

    _srows = []
    if _tags:
        for _f in sorted(glob.glob(os.path.join(_raw_abs, f"gpu_summary_{_SEL}_*.json"))):
            _s = json.load(open(_f))
            _kv = list(_s["kv_cache"]["seq_lens"].values())[-1]["reduction_pct"]
            _srows.append(
                f"| {_s['ratio']:.3f} | {_s['retention']['wt2_unpruned']['ppl']:.4f} | "
                f"{_s['retention']['wt2_pruned']['ppl']:.4f} | "
                f"{100 * _s['retention']['lambada_unpruned']['top1_accuracy']:.1f} | "
                f"{100 * _s['retention']['lambada_pruned']['top1_accuracy']:.1f} | {_kv:.2f} |"
            )

    CARDS2 = []
    if os.path.isdir(_raw_abs):
        for _p in sorted(glob.glob(os.path.join(_raw_abs, f"gpu_*_{_SEL}_*.json"))):
            _nm = os.path.basename(_p)
            if "30" in _nm or "29.2" in _nm:
                continue
            with open(_p, "rb") as _fh:
                CARDS2.append(mo.download(data=_fh.read(), filename=_nm))
    if os.path.isdir(_fig_abs):
        for _fig in ("latency", "kv"):
            _p = os.path.join(_fig_abs, f"gpu_{_fig}_{_SEL}.png")
            if os.path.exists(_p):
                with open(_p, "rb") as _fh:
                    CARDS2.append(mo.download(data=_fh.read(), filename=os.path.basename(_p)))

    mins2 = (time.perf_counter() - _t0) / 60
    _fail_md2 = (
        [mo.md("### ⚠️ ফেল হওয়া রান — রিপোর্ট এজেন্টকে দাও")]
        + [mo.accordion({f"⚠️ {_fn} (exit {_rc})": _err for _fn, _rc, _err in _fails})]
        if _fails else []
    )

    RESULT_SWEEP = mo.vstack(
        [mo.md(
            f"## ✅ FULL-RATIO SWEEP শেষ ({mins2:.1f} মিনিট) — model={_SEL} · "
            f"tags: {', '.join(_tags) if _tags else '—'}"
            + ("" if not _fails else " · ⚠️ কিছু রান ফেল")
        )]
        + ([mo.md(
            f"### 📊 {_SEL} — measured GPU matrix (rtx pro 6000, fp16)\n\n"
            "| ratio | WT2 base | WT2 pruned | LAMB base | LAMB pruned | KV % |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(_srows)
        )] if _srows else [])
        + ([mo.md("### 📥 download sweep artifacts (30% canonical বাদ)")]+ CARDS2
           if CARDS2 else [mo.md("_নতুন sweep আর্টিফ্যাক্ট নেই_")])
        + _fail_md2
        + [mo.md("### 📜 run logs"),
           mo.accordion({f"📜 {_t}": _log for _t, _log in _tails.items()})
           if _tails else mo.md("_কোনো রান হয়নি_")]
        + [UI_ALL],
    )
    return (RESULT_SWEEP,)


@app.cell
def _(RESULT_SWEEP):
    RESULT_SWEEP
    return


if __name__ == "__main__":
    app.run()
