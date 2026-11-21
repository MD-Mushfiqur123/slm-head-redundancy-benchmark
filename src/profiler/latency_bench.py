"""High-precision synchronized CPU/CUDA latency benchmarking."""
from __future__ import annotations

import time
from statistics import mean, stdev
from typing import Dict, List, Optional

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def prefill_latency_ms(
    model,
    input_ids: torch.Tensor,
    repeats: int = 8,
    warmup: int = 3,
) -> Dict[str, float]:
    """Timed single forward pass (prefill) over full context, ms/forward."""
    device = next(model.parameters()).device
    for _ in range(warmup):
        model(input_ids, use_cache=False)
    _sync(device)
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(input_ids, use_cache=False)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    return {"mean_ms": mean(times), "std_ms": stdev(times) if len(times) > 1 else 0.0}


@torch.no_grad()
def decode_latency_ms_per_token(
    model,
    input_ids: torch.Tensor,
    decode_steps: int = 24,
    repeats: int = 5,
    warmup: int = 2,
    past_key_values=None,
    use_cache: bool = True,
) -> Dict[str, float]:
    """Timed autoregressive decode: mean wall-clock milliseconds per newly
    generated token, including the incremental KV-cache writes."""
    device = next(model.parameters()).device
    logits = None
    for _ in range(warmup):
        out = model(input_ids, use_cache=use_cache)
        past = out.past_key_values if use_cache else None
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(decode_steps):
            out = model(next_tok, past_key_values=past, use_cache=use_cache)
            past = out.past_key_values if use_cache else None
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    _sync(device)
    times: List[float] = []
    for _ in range(repeats):
        out = model(input_ids, use_cache=use_cache)
        past = out.past_key_values if use_cache else None
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(decode_steps):
            out = model(next_tok, past_key_values=past, use_cache=use_cache)
            past = out.past_key_values if use_cache else None
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1e3 / decode_steps)
    return {
        "mean_ms_per_token": mean(times),
        "std_ms_per_token": stdev(times) if len(times) > 1 else 0.0,
    }


class LatencyBenchmark:
    """Convenience wrapper producing the Table-6 telemetry dictionaries."""

    def __init__(self, decode_steps: int = 24, repeats: int = 5, warmup: int = 2) -> None:
        self.decode_steps = decode_steps
        self.repeats = repeats
        self.warmup = warmup

    def run(self, model, tokenizer, prompt: str, batch_sizes=(1, 4, 8), seq_len: int = 256) -> Dict[int, Dict[str, float]]:
        device = next(model.parameters()).device
        results = {}
        for b in batch_sizes:
            inputs = tokenizer(
                [prompt] * b, return_tensors="pt", padding="max_length",
                max_length=seq_len, truncation=True,
            )
            results[b] = decode_latency_ms_per_token(
                model,
                inputs["input_ids"][:, -1:].to(device),
                decode_steps=self.decode_steps,
                repeats=self.repeats,
                warmup=self.warmup,
            )
            results[b]["prefill_ms"] = prefill_latency_ms(
                model, inputs["input_ids"][:, -seq_len:].to(device),
                repeats=self.repeats, warmup=self.warmup,
            )["mean_ms"]
        return results
