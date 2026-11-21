"""KV-cache footprint (Theorem 3) plus peak physical memory tracking."""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from typing import Dict, Optional

import numpy as np
import torch


def kv_cache_footprint_mb(
    num_layers: int,
    active_heads_per_layer: int,
    d_head: int,
    batch_size: int,
    seq_len: int,
    dtype_bytes: int = 2,
) -> float:
    """Theorem 3: M_KV = 2 * B * T * N_active * d_k * P bytes per layer,
    totalled over layers. Returns value in megabytes (MB = 2**20 bytes)."""
    bytes_total = (
        2
        * batch_size
        * seq_len
        * active_heads_per_layer
        * d_head
        * dtype_bytes
        * num_layers
    )
    return bytes_total / (1024.0 ** 2)


def kv_cache_reduction_mb(
    num_layers: int,
    num_heads: int,
    pruned_mask,
    d_head: int,
    batch_size: int,
    seq_len: int,
    dtype_bytes: int = 2,
) -> float:
    """Linear memory saved by excising heads (Theorem 3 delta)."""
    import numpy as np

    mask = np.asarray(mask if mask is not None else np.ones((num_layers, num_heads)))
    removed = int(num_layers * num_heads - mask.sum())
    bytes_total = 2 * batch_size * seq_len * removed * d_head * dtype_bytes
    return bytes_total / (1024.0 ** 2)


def peak_rss_mb() -> float:
    """Peak resident set size of the current process in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@torch.no_grad()
def _peak_hwm_mb() -> float:
    """VmHWM from /proc (fallback: ru_maxrss) — peak resident set, MB."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _reset_hwm() -> None:
    """Reset the peak-RSS high-water mark (kernel clears VmHWM on "5")."""
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5\n")
    except OSError:
        pass


def _kv_worker(payload: Dict) -> Dict:
    """Fresh-interpreter workload: load the model, run a T-token prefill with
    logits_to_keep=1 so activations stay negligible, and report the peak-RSS
    delta (VmHWM) attributable to the materialised KV cache."""
    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, payload["repo_root"])
    model = AutoModelForCausalLM.from_pretrained(
        payload["model_id"], attn_implementation="eager"
    )
    model.eval()
    if payload.get("pruned_heads"):
        from src.pruners.structured_pruner import excise_heads

        num_layers = payload["num_layers"]
        num_heads = payload["num_heads"]
        pruned = payload["pruned_heads"]
        mask = np.ones((num_layers, num_heads), dtype=np.float64)
        for layer_s, heads in pruned.items():
            mask[int(layer_s), list(heads)] = 0.0
        model = excise_heads(model, mask)
    tokenizer = AutoTokenizer.from_pretrained(payload["model_id"])
    seq_len = min(payload["seq_len"], payload["context_limit"])
    ids = tokenizer.encode(payload["prompt"])
    if not ids:
        ids = [ tokenizer.eos_token_id or 0 ]
    ids = (ids * (seq_len // len(ids) + 1))[:seq_len]
    input_ids = torch.tensor([ids], dtype=torch.long)
    gc.collect()
    with torch.no_grad():  # one-time library/kernel allocations, excluded from the delta
        try:
            model(input_ids[:, :8], use_cache=True, logits_to_keep=1)
        except TypeError:
            model(input_ids[:, :8], use_cache=True)
    gc.collect()
    _reset_hwm()
    base_hwm = _peak_hwm_mb()
    with torch.no_grad():
        try:
            model(input_ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            model(input_ids, use_cache=True)
    after_hwm = _peak_hwm_mb()
    return {"seq_len": seq_len, "rss_delta_mb": max(0.0, after_hwm - base_hwm)}


def empirical_kv_memory_mb(
    model,
    tokenizer,
    prompt: str,
    seq_len: int,
    batch_size: int = 1,
    pruned_heads: Optional[Dict[int, list]] = None,
) -> Optional[float]:
    """Measured peak-RSS delta of a fresh subprocess materialising a KV cache.

    Runs the workload in a new interpreter so the allocator caches of the
    parent process cannot mask the cache footprint. ``pruned_heads`` maps
    layer index -> list of excised head indices for physically pruned
    variants. Returns None if the measurement subprocess fails.
    """
    del tokenizer, batch_size  # the worker re-tokenises in a clean interpreter
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    context_limit = int(getattr(model.config, "max_position_embeddings", 1024) or 1024)
    payload = {
        "model_id": model.config._name_or_path,
        "prompt": prompt,
        "seq_len": seq_len,
        "repo_root": repo_root,
        "context_limit": context_limit,
        "num_layers": int(model.config.num_hidden_layers
                          if hasattr(model.config, "num_hidden_layers")
                          else model.config.n_layer),
        "num_heads": int(model.config.num_attention_heads
                         if hasattr(model.config, "num_attention_heads")
                         else model.config.n_head),
    }
    if pruned_heads:
        payload["pruned_heads"] = {str(int(k)): [int(h) for h in v]
                                   for k, v in pruned_heads.items() if len(v)}
    proc = subprocess.run(
        [sys.executable, "-m", "src.profiler.memory_tracker", json.dumps(payload)],
        capture_output=True, text=True, cwd=repo_root,
    )
    if proc.returncode != 0:
        return None
    for line in reversed(proc.stdout.strip().splitlines()):
        try:
            return float(json.loads(line)["rss_delta_mb"])
        except (ValueError, KeyError, TypeError):
            continue
    return None


def memory_report(model) -> Dict[str, float]:
    """Parameter memory in fp32/fp16 MB plus peak process RSS."""
    params = sum(p.numel() for p in model.parameters())
    return {
        "params_millions": params / 1e6,
        "weights_fp32_mb": params * 4 / (1024.0 ** 2),
        "weights_fp16_mb": params * 2 / (1024.0 ** 2),
        "peak_rss_mb": peak_rss_mb(),
    }


if __name__ == "__main__":
    _payload = json.loads(sys.argv[1])
    print(json.dumps(_kv_worker(_payload)), flush=True)
