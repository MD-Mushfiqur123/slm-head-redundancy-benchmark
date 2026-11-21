"""Theoretical FLOPs / parameter accounting for head pruning."""
from __future__ import annotations

from typing import Dict, List


def attention_flops_per_token(num_heads: int, d_model: int, d_head: int, seq_len: int) -> int:
    """Dense MHA FLOPs per token (multiply-accumulate counted as 2 ops):
    QKV projections (3 x 2 x d x d) + attention scores/values
    (2 x 2 x T x d_k x N_h) + output projection (2 x d x d)."""
    proj_qkv = 6 * d_model * d_model
    attn_scores = 2 * 2 * seq_len * d_head * num_heads
    attn_values = 2 * 2 * seq_len * d_head * num_heads
    proj_out = 2 * d_model * d_model
    return int(proj_qkv + attn_scores + attn_values + proj_out)


def attention_flops_pruned(
    active_heads_per_layer: List[int], d_model: int, d_head: int, seq_len: int
) -> int:
    """FLOPs/token with GQA-style head counts per layer."""
    total = 0
    for active in active_heads_per_layer:
        per_head_proj = 2 * d_model * d_head
        qkv = active * per_head_proj * 3
        scores = 2 * 2 * seq_len * d_head * active
        values = 2 * 2 * seq_len * d_head * active
        out = 2 * d_model * d_head * active
        total += int(qkv + scores + values + out)
    return total


def mla_parameter_millions_per_layer(
    num_heads: int, d_model: int, d_head: int
) -> float:
    """Attention projection parameter count per layer in millions."""
    return (4 * d_model * num_heads * d_head) / 1e6


def attention_parameter_millions(
    active_heads_per_layer: List[int], d_model: int, d_head: int
) -> float:
    return sum(
        4 * d_model * a * d_head for a in active_heads_per_layer
    ) / 1e6


def reduction_report(
    num_layers: int, num_heads: int, d_model: int, d_head: int, mask, seq_len: int = 512
) -> Dict[str, float]:
    """Full theoretical accounting for one mask."""
    import numpy as np

    mask = np.asarray(mask)
    if mask.shape != (num_layers, num_heads):
        raise ValueError("mask shape mismatch")
    active = [int(mask[li].sum()) for li in range(num_layers)]
    total_heads = num_layers * num_heads
    active_heads = int(sum(active))
    return {
        "total_heads": total_heads,
        "active_heads": active_heads,
        "pruned_heads": total_heads - active_heads,
        "pruning_ratio": (total_heads - active_heads) / total_heads,
        "attention_params_millions": attention_parameter_millions(active, d_model, d_head),
        "flops_per_token_attention": attention_flops_pruned(active, d_model, d_head, seq_len),
        "attention_share_pruned_pct": 100.0 * (total_heads - active_heads) / total_heads,
    }
