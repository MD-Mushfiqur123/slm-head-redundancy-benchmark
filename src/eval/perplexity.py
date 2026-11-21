"""Autoregressive sliding-window perplexity engine.

Windows are formed with ``max_length`` context and ``stride`` overlap; for
windows after the first, NLL is scored only on the newly revealed trailing
tokens so every token contributes exactly once. Returns
``exp(total_nll / total_tokens)`` plus wall-clock latency statistics.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence

import numpy as np
import torch


def _logits(out):
    """Support both HF ModelOutput objects and plain dicts."""
    import torch

    logits = getattr(out, "logits", None)
    if logits is None and isinstance(out, dict):
        logits = out.get("logits")
    return logits


def build_windows(
    seq: Sequence[int], max_length: int, stride: int
) -> List[Dict]:
    """Sliding windows over one tokenised sequence with non-recounted tails."""
    windows: List[Dict] = []
    if len(seq) <= max_length:
        windows.append({"tokens": list(seq), "score_from": 1})
        return windows
    pos = 0
    while True:
        end = pos + max_length
        window = seq[pos:end]
        if len(window) < 2:
            break
        if pos == 0:
            score_from = 1
        else:
            score_from = min(len(window), max_length - stride)
            if end >= len(seq):
                pass
        windows.append({"tokens": list(window), "score_from": score_from})
        if end >= len(seq):
            break
        pos += stride
    return windows


@torch.no_grad()
def evaluate_sliding_perplexity(
    model,
    sequences: Sequence[Sequence[int]],
    pad_id: int,
    batch_size: int = 8,
    stride: int = 256,
    max_length: int = 512,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """Exact sliding-window NLL over all evaluation sequences, batched."""
    was_training = model.training
    model.eval()
    all_windows: List[Dict] = []
    for seq in sequences:
        all_windows.extend(build_windows(list(seq), max_length=max_length, stride=stride))

    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()
    for i in range(0, len(all_windows), batch_size):
        chunk = all_windows[i : i + batch_size]
        maxlen = max(len(w["tokens"]) for w in chunk)
        ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        labels = torch.full((len(chunk), maxlen), -100, dtype=torch.long)
        for bi, w in enumerate(chunk):
            t = torch.tensor(w["tokens"], dtype=torch.long)
            ids[bi, : len(w["tokens"])] = t
            labels[bi, w["score_from"]: len(w["tokens"])] = t[
                w["score_from"] : len(w["tokens"])
            ]
        ids = ids.to(device)
        labels = labels.to(device)
        out = model(input_ids=ids, labels=labels, use_cache=False)
        logits = _logits(out)
        # chunked fp32 log-softmax (vocabulary is large; avoids full double
        # materialisation); totals accumulate in float64 for exactness
        shift_logits = logits[:, :-1, :]
        tgt = labels[:, 1:]
        mask = tgt != -100
        safe_tgt = tgt.clamp(min=0)
        chunk = 128
        n_pos = shift_logits.shape[1]
        for s in range(0, n_pos, chunk):
            e = min(s + chunk, n_pos)
            lp = torch.log_softmax(shift_logits[:, s:e, :].float(), dim=-1)
            m = mask[:, s:e]
            if not bool(m.any()):
                continue
            tok_nll = -lp.gather(-1, safe_tgt[:, s:e].unsqueeze(-1)).squeeze(-1)
            total_nll += float(tok_nll[m].double().sum().item())
            total_tokens += int(m.sum().item())
        del out, logits, shift_logits
    elapsed = time.perf_counter() - t0
    if was_training:
        model.train()
    ppl = math.exp(total_nll / total_tokens)
    return {
        "perplexity": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "num_sequences": len(sequences),
        "num_windows": len(all_windows),
        "eval_time_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
    }
