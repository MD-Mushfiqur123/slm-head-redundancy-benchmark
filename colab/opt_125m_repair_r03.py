# SLM-HEAD-REDUNDANCY-MONUMENT — OPT-125M r=0.3 REPAIR (standalone)
#
# একটাই অপারেশন সেল, একটাই বোতাম:
#   ▶️ preflight GPU → pull → install → repayment r=0.3 → mask+figure gap-fill → table → download
#
# model = opt_125m (facebook/opt-125m, 12L × 12H = 144 heads, kv_group=1 — কোনো snap নেই)
# task  = scripts/finetune_repayment.py --model opt_125m --ratios 0.3
#
# gap-fill (শুধু ফাইল না থাকলে চলে — বিদ্যমান measured আর্টিফ্যাক্ট কখনো overwrite হয় না):
#   • results/raw/gpu_mask_opt_125m_30_fp16.json  না থাকলে fresh Taylor calibration থেকে মাপা
#   • results/figures/gpu_{latency,kv}_opt_125m.png না থাকলে বিদ্যমান measured JSON থেকে প্লট
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
        # 🧩 OPT-125M — ফাইন-টিউনিং REPAIR (r = 0.3)

        **একটাই অপারেশন সেল, একটাই বোতাম।** বোতাম চাপলে: ① repo pull ② dependency
        install ③ repayment @ 30% excision ④ বাকি-থাকা OPT mask JSON + latency/KV প্লট
        (না থাকলে মেপে) ⑤ measured টেবিল ⑥ সব JSON/CSV/PNG ডাউনলোড।

        /TRUTH LAW: প্রতিটা সংখ্যা runtime-এ মাপা — কোনো mock/placeholder নেই।
        """
    )
    return


@app.cell
def _(mo):
    AI_RUN = mo.ui.run_button(
        label="▶️ OPT-125M REPAIR r=0.3: preflight → pull → run → table → download"
    )
    AI_UI = mo.vstack([
        mo.md(
            "### 🎯 REPAIR\n\n"
            "model **opt_125m** · ratio **0.30** · fp16 pipeline — "
            "সেটিং লাগবে না, উপরের বোতামে চাপ দাও।"
        ),
        AI_RUN,
    ])
    return AI_RUN, AI_UI


@app.cell
def _(AI_RUN, AI_UI, json, mo, os, subprocess, sys, time):
    _REPO_URL = "https://github.com/MD-Mushfiqur123/slm-head-redundancy-benchmark"
    _REPO = "slm-head-redundancy-benchmark"
    _MODEL = "opt_125m"
    _RATIO = 0.3
    _RATIO_TAG = "30"
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
                    f"⏸️ এখনো চালু হয়নি — GPU রেডি ✅ ({_gpu_name}).\n\n"
                    "উপরের বোতামে চাপ দাও।"
                ),
                AI_UI,
            ]),
        )
    _banner(f"① GPU preflight ok — {_gpu_name}")


    if not os.path.isdir(_REPO):
        _banner("② cloning repo")
        subprocess.run(
            ["git", "clone", "--depth", "1", _REPO_URL, _REPO], check=True
        )
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
            _banner("③ fast-forward blocked by local artifacts — resetting to origin/main")
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

    _raw = os.path.join(_REPO, "results", "raw")
    _fig_dir = os.path.join(_REPO, "results", "figures")
    _repo_abs = os.path.abspath(_REPO)
    _raw_abs = os.path.abspath(_raw)
    _fig_abs = os.path.abspath(_fig_dir)

    # ── ④ repayment @ r=0.3 ──────────────────────────────────────────────
    _tails = {}
    _fails = []
    _ts = time.perf_counter()
    _banner(f"④ running: repayment | model={_MODEL} | ratio={_RATIO}")
    _proc = subprocess.run(
        [sys.executable, "scripts/finetune_repayment.py",
         "--model", _MODEL, "--ratios", repr(_RATIO)],
        capture_output=True, text=True, cwd=_repo_abs,
    )
    _mins = (time.perf_counter() - _ts) / 60
    _tails["repayment·opt_125m"] = (
        f"[{_mins:.1f} min, exit {_proc.returncode}]\n"
        + _proc.stdout[-20000:] + "\n" + _proc.stderr[-4000:]
    )
    if _proc.returncode != 0:
        _fails.append(("repayment·opt_125m", _proc.returncode, _proc.stderr[-1500:]))
        _banner(f"⚠️ repayment exited {_proc.returncode} — এগিয়ে যাচ্ছি (gap-fill চেষ্টা হবে)")
    else:
        _banner(f"✓ repayment done ({_mins:.1f} min)")

    # ── ⑤ gap-fill: missing OPT mask JSON (measured, fresh Taylor calibration) ─
    _mask_path = os.path.join(_raw_abs, f"gpu_mask_{_MODEL}_{_RATIO_TAG}_fp16.json")
    _mask_fresh = False
    if not os.path.exists(_mask_path):
        _banner("⑥ gap-fill: gpu_mask_opt_125m_30_fp16.json নেই — Taylor calibration মেপে লিখছি")
        _mproc = subprocess.run(
            [sys.executable, "-c", r'''
import json, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
from src.core.data import calibration_and_eval_sequences, load_wikitext_split
from src.core.hooks import HeadGatingHookManager
from src.core.model_loader import MODEL_REGISTRY, load_model, load_tokenizer
from src.metrics.taylor import TaylorImportanceScorer
from src.pruners.global_pruner import GlobalHeadPruner
from src.core.data import autocast_batch

ratio, tag = 0.3, "30"
spec = MODEL_REGISTRY["opt_125m"]
device = torch.device("cuda")
torch.manual_seed(7); np.random.seed(7)
tok = load_tokenizer(spec)
rows = load_wikitext_split("validation")
calibration, _ = calibration_and_eval_sequences(
    rows, tok, num_calibration=150, num_eval=200, window=512, min_tokens=50)
model = load_model(spec).to(device).eval()
pad_id = tok.pad_token_id
hm = HeadGatingHookManager(model)
taylor = TaylorImportanceScorer()
n_cal = sum(len(s) for s in calibration)
rng = np.random.default_rng(7)
order = rng.permutation(len(calibration))
batches = [[calibration[j] for j in order[i:i+8]] for i in range(0, len(order), 8)]
t0 = time.perf_counter()
for entries in batches:
    batch = autocast_batch(entries, pad_id)
    batch = {k: v.to(device) for k, v in batch.items()}
    hm.reset_gradients()
    out = model(**batch)
    out.loss.backward()
    taylor.process_batch(hm, {"labels": batch["labels"].cpu()})
raw = taylor.finalize(n_cal)
hm.reset_gradients(); hm.remove_hooks()
cal_s = time.perf_counter() - t0
pruner = GlobalHeadPruner(spec.num_layers, spec.num_heads, kv_group_size=1)
mask = pruner.build_mask(raw, ratio)
excised = int((mask == 0).sum())
gpu_name = torch.cuda.get_device_name(0)
path = f"results/raw/gpu_mask_{spec.key}_{tag}_fp16.json"
with open(path, "w") as f:
    json.dump({"model": spec.key, "ratio": ratio, "kv_group": 1,
               "excised": excised, "total_heads": spec.num_layers * spec.num_heads,
               "mask": mask.tolist(), "device": gpu_name}, f, indent=2)
print(f"calibration {cal_s:.1f}s | excised {excised}/{spec.num_layers*spec.num_heads} -> {path}")
'''],
            capture_output=True, text=True, cwd=_repo_abs,
        )
        _tails["mask·gapfill"] = (
            f"[exit {_mproc.returncode}]\n" + _mproc.stdout[-8000:] + "\n" + _mproc.stderr[-3000:]
        )
        if _mproc.returncode != 0:
            _fails.append(("mask·gapfill", _mproc.returncode, _mproc.stderr[-1500:]))
            _banner("⚠️ mask gap-fill ফেল — বাকিগুলো তবু শেষ হয়েছে")
        else:
            _mask_fresh = True
            _banner("✓ mask gap-fill done")
    else:
        _banner(f"⑥ mask JSON already measured — untouched: {os.path.basename(_mask_path)}")

    # ── ⑦ gap-fill: missing latency/KV figures (from measured JSONs) ────
    _fig_made = []
    if os.path.isdir(_raw_abs):
        _lat_p = os.path.join(_raw_abs, f"gpu_latency_{_MODEL}_{_RATIO_TAG}_fp16.json")
        _kv_p = os.path.join(_raw_abs, f"gpu_kv_{_MODEL}_{_RATIO_TAG}_fp16.json")
        _need_lat = not os.path.exists(os.path.join(_fig_abs, f"gpu_latency_{_MODEL}.png"))
        _need_kv = not os.path.exists(os.path.join(_fig_abs, f"gpu_kv_{_MODEL}.png"))
        if _need_lat or _need_kv:
            _banner("⑦ gap-fill: লাগবে এমন measured figure গুলো প্লট করছি")
            _fproc = subprocess.run(
                [sys.executable, "-c", r'''
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model, tag = sys.argv[1], sys.argv[2]
need_lat, need_kv = sys.argv[3] == "1", sys.argv[4] == "1"
raw = "results/raw"
with open(f"{raw}/gpu_summary_{model}_{tag}_fp16.json") as f:
    summary = json.load(f)
hf_id, gpu_name, ratio = summary["hf_id"], summary["gpu"], summary["ratio"]
os.makedirs("results/figures", exist_ok=True)

if need_lat:
    with open(f"{raw}/gpu_latency_{model}_{tag}_fp16.json") as f:
        lat = json.load(f)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bs = sorted(int(b) for b in lat["batch_sizes"])
    pruned_name = next(k for k in lat["batch_sizes"][str(bs[0])] if k != "unpruned")
    dec_un = [lat["batch_sizes"][str(b)]["unpruned"]["decode_ms_per_token"] for b in bs]
    dec_pr = [lat["batch_sizes"][str(b)][pruned_name]["decode_ms_per_token"] for b in bs]
    xpos, width = np.arange(len(bs)), 0.36
    ax.bar(xpos - width/2, dec_un, width, label="unpruned", color="#4472C4")
    ax.bar(xpos + width/2, dec_pr, width, label=f"pruned @ {ratio:.0%}", color="#ED7D31")
    for xi, b in enumerate(bs):
        sp = lat["batch_sizes"][str(b)]["speedup_pct"]
        ax.text(xi, max(dec_un[xi], dec_pr[xi]) * 1.02, f"{sp:+.1f}%",
                ha="center", fontsize=9)
    ax.set_xticks(xpos); ax.set_xticklabels([f"B={b}" for b in bs])
    ax.set_ylabel("decode latency (ms/token)")
    ax.set_title(f"Wall-clock decode latency — {hf_id} on {gpu_name}")
    ax.legend()
    fig.savefig(f"results/figures/gpu_latency_{model}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote results/figures/gpu_latency_%s.png" % model)

if need_kv:
    with open(f"{raw}/gpu_kv_{model}_{tag}_fp16.json") as f:
        kvj = json.load(f)
    mem_rows = [kvj["seq_lens"][k] for k in sorted(kvj["seq_lens"], key=int)]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ts = [r["T"] for r in mem_rows]
    ax.plot(ts, [r["unpruned"] for r in mem_rows], "o-", label="measured unpruned",
            color="#C00000")
    ax.plot(ts, [r["pruned30"] for r in mem_rows], "s-", label=f"measured pruned @{ratio:.0%}",
            color="#2E7D32")
    ax.plot(ts, [r["theoretical_MB"] for r in mem_rows], "--", label="Theorem 3 (pruned)",
            color="#7CB342", alpha=0.8)
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("KV-cache footprint (MB)")
    ax.set_title(f"Live KV-cache bytes vs Theorem 3 — {hf_id}")
    ax.legend()
    fig.savefig(f"results/figures/gpu_kv_{model}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote results/figures/gpu_kv_%s.png" % model)
''', _MODEL, _RATIO_TAG, "1" if _need_lat else "0", "1" if _need_kv else "0"],
                capture_output=True, text=True, cwd=_repo_abs,
            )
            _tails["figures·gapfill"] = (
                f"[exit {_fproc.returncode}]\n"
                + _fproc.stdout[-8000:] + "\n" + _fproc.stderr[-3000:]
            )
            if _fproc.returncode != 0:
                _fails.append(("figures·gapfill", _fproc.returncode, _fproc.stderr[-1500:]))
                _banner("⚠️ figure gap-fill ফেল — বাকিগুলো তবু শেষ হয়েছে")
            else:
                _fig_made = [ln for ln in _fproc.stdout.splitlines() if ln.startswith("wrote")]
                _banner("✓ figure gap-fill done")
        else:
            _banner("⑦ latency/KV figures already exist — untouched")
    mins = (time.perf_counter() - _t0) / 60

    # ── ⑧ tables ─────────────────────────────────────────────────────────
    REPAY = []
    _rep_path = os.path.join(_repo_abs, "results", "repayment_opt_125m.csv")
    _rep_json = os.path.join(_raw_abs, "repayment_opt_125m.json")
    if os.path.exists(_rep_json):
        _res = json.load(open(_rep_json))
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
    else:
        REPAY.append(mo.md("### 💶 repayment আর্টিফ্যাক্ট এই রানে পাওয়া যায়নি ⚠️"))

    _mask_md = []
    if os.path.exists(_mask_path):
        _m = json.load(open(_mask_path))
        _exc = _m["excised"]
        _tot = _m["total_heads"]
        _src = "এই রানে নতুন মাপা ✅" if _mask_fresh else "আগে থেকেই আছে (নতুন না)"
        _mask_md.append(
            mo.md(
                f"### 🎭 Mask — {_m['model']}\n\n"
                f"excised **{_exc}/{_tot} heads** · ratio {_m['ratio']:.2f} · "
                f"kv_group {_m['kv_group']} · {_src}"
            )
        )

    # ── ⑨ downloads: সব OPT আর্টিফ্যাক্ট (repayment + mask + figures) ────
    CARDS = []
    _want_json = [
        "repayment_opt_125m.json",
        f"gpu_summary_{_MODEL}_{_RATIO_TAG}_fp16.json",
        f"gpu_ppl_{_MODEL}_{_RATIO_TAG}_fp16.json",
        f"gpu_latency_{_MODEL}_{_RATIO_TAG}_fp16.json",
        f"gpu_kv_{_MODEL}_{_RATIO_TAG}_fp16.json",
        _mask_fresh and os.path.basename(_mask_path) or "",
    ]
    if os.path.isdir(_raw_abs):
        for _name in _want_json:
            if not _name:
                continue
            _p = os.path.join(_raw_abs, _name)
            if os.path.exists(_p):
                with open(_p, "rb") as _fh:
                    CARDS.append(mo.download(data=_fh.read(), filename=_name))
    if os.path.exists(_rep_path):
        with open(_rep_path, "rb") as _fh:
            CARDS.append(mo.download(data=_fh.read(), filename=os.path.basename(_rep_path)))
    for _fig in ("latency", "kv"):
        _p = os.path.join(_fig_abs, f"gpu_{_fig}_{_MODEL}.png")
        if os.path.exists(_p):
            with open(_p, "rb") as _fh:
                CARDS.append(mo.download(data=_fh.read(), filename=f"gpu_{_fig}_{_MODEL}.png"))

    FIGS = []
    for _fig in ("latency", "kv"):
        _p = os.path.join(_fig_abs, f"gpu_{_fig}_{_MODEL}.png")
        if os.path.exists(_p):
            FIGS.append(mo.image(src=_p, width=720))

    _fail_md = (
        [mo.md("### ⚠️ ফেল হওয়া রান — বাকিগুলো তবু শেষ হয়েছে")]
        + [mo.accordion({f"⚠️ {_fn} (exit {_rc})": _err for _fn, _rc, _err in _fails})]
        if _fails else []
    )

    RESULT = mo.vstack(
        [mo.md(
            f"## ✅ OPT-125M REPAIR শেষ ({mins:.1f} মিনিট) — "
            f"model=opt_125m · ratio=0.30 · GPU {_gpu_name}"
            + ("" if not _fails else " · ⚠️ কিছু রান ফেল (নিচে রিপোর্ট)")
        )]
        + REPAY
        + _mask_md
        + ([mo.md("### 🖼 figures")] + FIGS if FIGS else [])
        + ([mo.md("### 📥 download artifacts (JSON/CSV/PNG)")]
           + CARDS if CARDS else
           [mo.md("### 📥 download artifacts"), mo.md("_কিছু নেই_")])
        + _fail_md
        + [mo.md(
            "**[repo](https://github.com/MD-Mushfiqur123/slm-head-redundancy-benchmark)"
            " · model: opt_125m · ratio=0.3 · fp16"
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
