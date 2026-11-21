"""Physical weight surgery for exact KV-cache / FLOPs reduction.

Excising a head is implemented by deleting its parameter rows/columns from
the QKV and output projections (and matching bias slices). For an ablated
head (gate = 0) the surgery is mathematically identity-preserving: the
pruned model's logits equal the gated model's logits exactly.
"""
from __future__ import annotations

import copy
from typing import List

import torch
import torch.nn as nn

from ..core.model_loader import AttentionSlicing, locate_attentions


def _replace_parameter(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    old = getattr(module, name)
    new = nn.Parameter(tensor.clone().contiguous(), requires_grad=old.requires_grad)
    setattr(module, name, new)


def _conv1d_param(module: nn.Module, attr: str):
    t = getattr(module, attr)
    if isinstance(t, torch.nn.Parameter):
        return t
    return t


@torch.no_grad()
def excise_heads(
    model: nn.Module,
    mask,
    slicing: List[AttentionSlicing] | None = None,
) -> nn.Module:
    """Return a new model with heads marked 0 in ``mask`` physically removed
    from all attention projections. Returns a model whose attention output
    equals the gated (mask-hooked) original model exactly.

    GQA guard: when key/value heads are shared by several query heads
    (num_kv_heads < num_heads), a shared K/V block is removed only once every
    query head referencing it has been pruned.
    """
    m = torch.as_tensor(mask).float()
    if slicing is None:
        slicing = locate_attentions(model)
    if m.shape != (len(slicing), slicing[0].num_heads):
        raise ValueError("mask shape mismatch with located slicing")

    pruned = copy.deepcopy(model)
    pruned.eval()
    pruned_slices = locate_attentions(pruned)

    heads_per_layer: List[int] = []
    for li, sl in enumerate(pruned_slices):
        keep = [h for h in range(sl.num_heads) if float(m[li, h]) > 0]
        heads_per_layer.append(len(keep))
        if len(keep) == sl.num_heads:
            continue
        if sl.is_conv1d:
            _excise_gpt2_layer(sl, keep)
        else:
            _excise_llama_layer(sl, keep)

    if hasattr(pruned, "config"):
        pruned.config._pruned_heads_per_layer = heads_per_layer
    return pruned


def _excise_gpt2_layer(sl: AttentionSlicing, keep: list) -> None:
    attn_conv = sl.qkv_param      # Conv1D: weight [d, 3d]
    out_conv = sl.o_proj          # Conv1D: weight [d, d]
    d = out_conv.weight.shape[0]
    dk = sl.d_head
    dev = out_conv.weight.device
    dt = out_conv.weight.dtype

    if not keep:
        # the whole layer's attention is excised: the output projection is
        # zeroed so attention contributes exactly nothing (identity to gating)
        _replace_parameter(out_conv, "weight", torch.zeros_like(out_conv.weight.detach()))
        if out_conv.bias is not None:
            _replace_parameter(out_conv, "bias", torch.zeros_like(out_conv.bias.detach()))
        return

    w = attn_conv.weight.detach()  # [d, 3d]
    b = attn_conv.bias.detach() if attn_conv.bias is not None else None
    keep_q = torch.cat(
        [torch.arange(h * dk, (h + 1) * dk, device=dev) for h in keep]
    )
    keep_k = keep_q + d
    keep_v = keep_q + 2 * d
    new_w = torch.cat([w[:, keep_q], w[:, keep_k], w[:, keep_v]], dim=1)
    _replace_parameter(attn_conv, "weight", new_w.to(dt))
    attn_conv.nf = new_w.shape[1]
    if b is not None:
        new_b = torch.cat([b[keep_q], b[keep_k], b[keep_v]])
        _replace_parameter(attn_conv, "bias", new_b.to(dt))

    wo = out_conv.weight.detach()  # [768 in, 768 out] -> rows = input features
    new_wo = wo[keep_q, :]
    _replace_parameter(out_conv, "weight", new_wo.to(dt))

    attn = sl.attn_module
    attn.split_size = len(keep) * dk
    attn.num_heads = len(keep)
    attn.head_dim = dk
    attn.embed_dim = attn.split_size


def _excise_llama_layer(sl: AttentionSlicing, keep: list) -> None:
    if not keep:
        _replace_parameter(
            sl.o_proj, "weight", torch.zeros_like(sl.o_proj.weight.detach())
        )
        if sl.o_proj.bias is not None:
            _replace_parameter(
                sl.o_proj, "bias", torch.zeros_like(sl.o_proj.bias.detach())
            )
        return
    q_proj = sl.qkv_param["q"]    # Linear: weight [nh*dk, d]
    k_proj = sl.qkv_param["k"]    # Linear: weight [nkv*dk, d]
    v_proj = sl.qkv_param["v"]
    o_proj = sl.o_proj            # Linear: weight [d, nh*dk]
    dk = sl.d_head
    dt = o_proj.weight.dtype

    keep_rows = torch.cat([torch.arange(h * dk, (h + 1) * dk) for h in keep])
    _replace_parameter(q_proj, "weight", q_proj.weight.detach()[keep_rows].to(dt))
    if q_proj.bias is not None:
        _replace_parameter(q_proj, "bias", q_proj.bias.detach()[keep_rows].to(dt))

    nkv = sl.num_kv_heads
    group = sl.num_heads // nkv
    keep_q_set = set(keep)
    keep_kv = [
        kv for kv in range(nkv)
        if any(h in keep_q_set for h in range(kv * group, (kv + 1) * group))
    ]
    keep_kv_rows = torch.cat([torch.arange(kv * dk, (kv + 1) * dk) for kv in keep_kv])
    _replace_parameter(k_proj, "weight", k_proj.weight.detach()[keep_kv_rows].to(dt))
    _replace_parameter(v_proj, "weight", v_proj.weight.detach()[keep_kv_rows].to(dt))
    if k_proj.bias is not None:
        _replace_parameter(k_proj, "bias", k_proj.bias.detach()[keep_kv_rows].to(dt))
    if v_proj.bias is not None:
        _replace_parameter(v_proj, "bias", v_proj.bias.detach()[keep_kv_rows].to(dt))

    keep_cols = keep_rows
    _replace_parameter(o_proj, "weight", o_proj.weight.detach()[:, keep_cols].to(dt))

    attn = sl.attn_module
    attn.num_heads = len(keep)
    attn.num_key_value_heads = len(keep_kv)
    for alias, value in (("nh", len(keep)), ("nkv", len(keep_kv))):
        if hasattr(attn, alias):
            setattr(attn, alias, value)
    if hasattr(attn, "num_key_value_groups"):
        attn.num_key_value_groups = len(keep) // len(keep_kv)
    if len(keep) % len(keep_kv) != 0:
        raise ValueError(
            "Non-uniform KV group sizes after pruning; masks must prune whole "
            "KV groups for GQA architectures"
        )
