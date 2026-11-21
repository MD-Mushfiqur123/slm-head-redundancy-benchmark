"""Model architecture registry and checkpoint loading utilities.

Supports GPT-2 (Post-LN, absolute positional embeddings, Conv1D attention
projections) and LLaMA-family sub-billion models (Pre-LN, RoPE, SwiGLU) such
as HuggingFaceTB/SmolLM-135M.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    """Static architecture description for one benchmark model."""

    key: str
    hf_id: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    d_model: int
    d_head: int
    arch_family: str
    position_strategy: str
    activation: str
    norm_style: str
    params_millions: float = 0.0


GPT2_SMALL = ModelSpec(
    key="gpt2_small",
    hf_id="gpt2",
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    d_model=768,
    d_head=64,
    arch_family="gpt2",
    position_strategy="absolute",
    activation="gelu_new",
    norm_style="post_ln",
)
GPT2_MEDIUM = ModelSpec(
    key="gpt2_medium",
    hf_id="gpt2-medium",
    num_layers=24,
    num_heads=16,
    num_kv_heads=16,
    d_model=1024,
    d_head=64,
    arch_family="gpt2",
    position_strategy="absolute",
    activation="gelu_new",
    norm_style="post_ln",
)
SMOLLM_135M = ModelSpec(
    key="smollm_135m",
    hf_id="HuggingFaceTB/SmolLM-135M",
    num_layers=30,
    num_heads=9,
    num_kv_heads=3,
    d_model=576,
    d_head=64,
    arch_family="llama",
    position_strategy="rope",
    activation="swiglu",
    norm_style="pre_rmsnorm",
)
OPT_125M = ModelSpec(
    key="opt_125m",
    hf_id="facebook/opt-125m",
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    d_model=768,
    d_head=64,
    arch_family="opt",
    position_strategy="learned",
    activation="relu",
    norm_style="pre_ln",
)
PYTHIA_160M = ModelSpec(
    key="pythia_160m",
    hf_id="EleutherAI/pythia-160m",
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    d_model=768,
    d_head=64,
    arch_family="gpt_neox",
    position_strategy="rope",
    activation="gelu",
    norm_style="pre_ln",
)
PYTHIA_410M = ModelSpec(
    key="pythia_410m",
    hf_id="EleutherAI/pythia-410m",
    num_layers=24,
    num_heads=16,
    num_kv_heads=16,
    d_model=1024,
    d_head=64,
    arch_family="gpt_neox",
    position_strategy="rope",
    activation="gelu",
    norm_style="pre_ln",
)
QWEN2_05B = ModelSpec(
    key="qwen2_05b",
    hf_id="Qwen/Qwen2.5-0.5B",
    num_layers=24,
    num_heads=14,
    num_kv_heads=2,
    d_model=896,
    d_head=64,
    arch_family="llama",
    position_strategy="rope",
    activation="swiglu",
    norm_style="pre_rmsnorm",
)

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        GPT2_SMALL, GPT2_MEDIUM, SMOLLM_135M,
        OPT_125M, PYTHIA_160M, PYTHIA_410M, QWEN2_05B,
    )
}


@dataclass
class AttentionSlicing:
    """Per-layer description of how attention heads map onto projection
    parameter tensors, enabling both output gating and physical weight
    surgery.

    Attributes:
        o_proj: the output projection module (input = concatenated heads).
        d_head: per-head value dimension.
        num_heads: number of query heads.
        num_kv_heads: number of key/value heads (1 when MKV disabled).
        head_dim_slices: for each query head h, the contiguous column slice of
            the o_proj input belonging to that head.
        qkv_param: the fused or separate QKV parameter module, or None.
        qkv_layout: mapping head index -> {'q': (col_range), 'k': ..., 'v': ...}
        is_conv1d: True when o_proj/qkv params store weights as (in, out).
        fused_qkv: True when qkv_param is a single interleaved GPT-NeoX module.
    """

    o_proj: nn.Module
    d_head: int
    num_heads: int
    num_kv_heads: int
    is_conv1d: bool
    qkv_param: Optional[nn.Module]
    layout: List[Dict[str, Optional[slice]]]
    attn_module: Optional[nn.Module] = None
    fused_qkv: bool = False


def _gpt2_head_slices(layer: nn.Module) -> AttentionSlicing:
    attn = layer.attn
    d = attn.embed_dim
    nh = attn.num_heads
    dk = attn.head_dim
    layout = []
    for h in range(nh):
        layout.append(
            {
                "q": slice(h * dk, (h + 1) * dk),
                "k": slice(d + h * dk, d + (h + 1) * dk),
                "v": slice(2 * d + h * dk, 2 * d + (h + 1) * dk),
            }
        )
    return AttentionSlicing(
        o_proj=attn.c_proj,
        d_head=dk,
        num_heads=nh,
        num_kv_heads=nh,
        is_conv1d=True,
        qkv_param=attn.c_attn,
        layout=layout,
        attn_module=attn,
    )


def _llama_head_slices(layer: nn.Module) -> AttentionSlicing:
    attn = layer.self_attn
    nh = attn.config.num_attention_heads
    nkv = getattr(attn.config, "num_key_value_heads", None) or nh
    dk = attn.head_dim
    qkv = {"q": attn.q_proj, "k": attn.k_proj, "v": attn.v_proj}
    o_proj = attn.o_proj if hasattr(attn, "o_proj") else attn.out_proj

    def col_slice(head: int) -> slice:
        return slice(head * dk, (head + 1) * dk)

    def kv_slice(head: int) -> tuple:
        group = attn.config.num_attention_heads // nkv
        return head // group, col_slice(head // group)

    layout = []
    for h in range(nh):
        kv_head, kv_cols = kv_slice(h)
        layout.append(
            {
                "q": col_slice(h),
                "k": (kv_head, kv_cols),
                "v": (kv_head, kv_cols),
            }
        )
    return AttentionSlicing(
        o_proj=o_proj,
        d_head=dk,
        num_heads=nh,
        num_kv_heads=nkv,
        is_conv1d=False,
        qkv_param=qkv,
        layout=layout,
        attn_module=attn,
    )


def _gpt_neox_head_slices(layer: nn.Module) -> AttentionSlicing:
    """Head slicing for GPT-NeoX (Pythia): a single fused QKV Linear whose
    rows are per-head interleaved blocks of [q k v], each of size d_head.
    Pythia has no GQA, so each query head owns its own K/V row block."""
    attn = layer.attention
    nh = attn.config.num_attention_heads
    nkv = getattr(attn.config, "num_key_value_heads", None) or nh
    if nkv != nh:
        raise ValueError("fused GPT-NeoX slicing assumes full MHA (no GQA)")
    dk = attn.head_size
    layout = [
        {
            "q": slice(h * 3 * dk, (h * 3 + 1) * dk),
            "k": slice((h * 3 + 1) * dk, (h * 3 + 2) * dk),
            "v": slice((h * 3 + 2) * dk, (h + 1) * 3 * dk),
        }
        for h in range(nh)
    ]
    return AttentionSlicing(
        o_proj=attn.dense,
        d_head=dk,
        num_heads=nh,
        num_kv_heads=nkv,
        is_conv1d=False,
        qkv_param=attn.query_key_value,
        layout=layout,
        attn_module=attn,
        fused_qkv=True,
    )


def locate_layers(model: nn.Module) -> List[nn.Module]:
    """Return the ordered decoder layer list for any supported architecture:
    GPT-2 (transformer.h), GPT-NeoX/Pythia (gpt_neox.layers), LLaMA family
    (model.layers) and OPT (model.decoder.layers)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return list(inner.layers)
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
            return list(inner.decoder.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError(
        "Unrecognised decoder hierarchy: no transformer.h / gpt_neox.layers "
        "/ model.layers / model.decoder.layers"
    )


def locate_attentions(model: nn.Module) -> List[AttentionSlicing]:
    """Build head-slicing metadata for every decoder layer."""
    layers = locate_layers(model)
    first = layers[0]
    if hasattr(first, "attn") and hasattr(first.attn, "c_proj"):
        return [_gpt2_head_slices(layer) for layer in layers]
    layer_attn = getattr(first, "self_attn", None) or getattr(first, "attention", None)
    if layer_attn is not None and hasattr(layer_attn, "query_key_value"):
        return [_gpt_neox_head_slices(layer) for layer in layers]
    if layer_attn is not None and hasattr(layer_attn, "k_proj"):
        return [_llama_head_slices(layer) for layer in layers]
    raise ValueError("Unrecognised attention module structure")


def load_model(
    spec: ModelSpec,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> nn.Module:
    """Load a causal LM checkpoint in evaluation mode.

    All weights stay frozen except the token embedding, which is kept a
    differentiable leaf so that backward passes still populate activation
    gradients (head-sensitivity telemetry) without materialising parameter
    gradients for the full network."""
    import torch.nn as nn

    from transformers import AutoModelForCausalLM, AutoConfig

    torch.manual_seed(seed)
    config = AutoConfig.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        config=config,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for param in module.parameters():
                param.requires_grad_(True)
    if hasattr(model, "config") and model.config.pad_token_id is None:
        model.config.pad_token_id = (
            model.config.eos_token_id if model.config.eos_token_id is not None else 0
        )
    return model


def load_tokenizer(spec: ModelSpec):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def coupon_shuffle_mask(
    num_layers: int,
    num_heads: int,
    ratio: float,
    generator: np.random.Generator,
    kv_group_size: int = 1,
) -> np.ndarray:
    """Uniform random binary head mask with round(L*Nh*ratio) zeros.
    For GQA models (kv_group_size > 1) whole KV groups are sampled."""
    total = num_layers * num_heads
    num_pruned = int(round(total * ratio))
    mask = np.ones((num_layers, num_heads), dtype=np.float32)
    if kv_group_size <= 1:
        idx = generator.permutation(total)[:num_pruned]
        for flat in idx:
            mask[flat // num_heads, flat % num_heads] = 0.0
        return mask
    gs = kv_group_size
    num_groups = num_layers * (num_heads // gs)
    num_groups_pruned = num_pruned // gs
    idx = generator.permutation(num_groups)[:num_groups_pruned]
    for flat in idx:
        li = flat // (num_heads // gs)
        g0 = (flat % (num_heads // gs)) * gs
        mask[li, g0 : g0 + gs] = 0.0
    return mask
