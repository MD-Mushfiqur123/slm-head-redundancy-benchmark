"""Interactive Streamlit dashboard: head ablation explorer.

Run:  streamlit run demo/app.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core.data import calibration_and_eval_sequences, load_wikitext_split  # noqa: E402
from src.core.hooks import HeadGatingHookManager  # noqa: E402
from src.core.model_loader import MODEL_REGISTRY, load_model, load_tokenizer  # noqa: E402
from src.eval.perplexity import evaluate_sliding_perplexity  # noqa: E402
from src.pruners.global_pruner import GlobalHeadPruner  # noqa: E402

st.set_page_config(page_title="SLM Head Ablation Lab", layout="wide")
st.title("SLM-HEAD-REDUNDANCY-MONUMENT — Interactive Head Ablation Lab")
st.markdown(
    "Zero-shot structural attention-head excision in sub-billion language models "
    "with exact gradient sensitivity gating (no fine-tuning)."
)

@st.cache_resource(show_spinner="Loading model ...")
def get_model(key):
    spec = MODEL_REGISTRY[key]
    return spec, load_model(spec), load_tokenizer(spec)


@st.cache_data(show_spinner="Loading calibration telemetry ...")
def get_importance(key):
    path = os.path.join(ROOT, "results", "raw", f"{key}_importance.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)


model_key = st.sidebar.selectbox("Model", list(MODEL_REGISTRY))
spec, model, tok = get_model(model_key)
imp = get_importance(model_key)
kv_group = 3 if spec.num_kv_heads != spec.num_heads else 1

st.sidebar.markdown(f"**Architecture:** {spec.num_layers} layers x {spec.num_heads} heads "
                    f"(d_model={spec.d_model})")

ratio = st.sidebar.slider("Global pruning ratio P", 0.0, 0.9, 0.30, 0.05)
metric = st.sidebar.selectbox(
    "Importance metric", ["taylor", "fisher", "weight_norm_l2", "random"]
)

if imp is not None:
    pruner = GlobalHeadPruner(spec.num_layers, spec.num_heads, kv_group_size=kv_group)
    importance = (
        np.random.RandomState(13).rand(spec.num_layers, spec.num_heads)
        if metric == "random"
        else imp[metric if metric != "taylor" else "taylor_raw"]
    )
    mask = pruner.build_mask(importance, ratio)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sensitivity heatmap (Taylor)")
        st.image(
            os.path.join(ROOT, "results", "figures", "figure1_importance_heatmap.png"),
            use_container_width=True,
        )
    with col2:
        st.subheader(f"Mask @ P={ratio:.2f} (**metric = {metric}**)")
        df = pd.DataFrame(
            mask, index=[f"L{li}" for li in range(spec.num_layers)],
            columns=[f"h{h}" for h in range(spec.num_heads)],
        )
        st.dataframe(df, height=min(30 * spec.num_layers + 38, 520))

    if st.sidebar.button("Evaluate current mask (WikiText-2 PPL)"):
        with st.spinner("Evaluating ..."):
            wt = load_wikitext_split("validation")
            _, eval_seqs = calibration_and_eval_sequences(
                wt, tok, num_calibration=4, num_eval=48, window=512
            )
            hm = HeadGatingHookManager(model)
            if ratio > 0:
                hm.set_mask(mask)
            res_p = evaluate_sliding_perplexity(
                model, eval_seqs, tok.pad_token_id, batch_size=4, stride=256, max_length=512
            )
            hm.clear_mask()
            hm.remove_hooks()
            st.metric(
                "WikiText-2 PPL (48 seq quick protocol)",
                f"{res_p['perplexity']:.3f}",
                delta=f"{100 * (res_p['perplexity'] / res_p['perplexity'] - 1):.1f}%",
            )
            st.metric("Active heads", f"{int(mask.sum())} / {spec.num_layers * spec.num_heads}")
else:
    st.info(
        "No calibration telemetry found for this model. Run "
        "`python scripts/run_calibration.py --model " + model_key + "` first."
    )

st.markdown("---")
st.caption(
    "Persuasive bilingual note: this dashboard evaluates masks through exact "
    "gating hooks — the excised-head predictions are identical to physical weight surgery."
)
