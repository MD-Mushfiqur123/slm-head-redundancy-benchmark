"""Zero-shot LAMBADA target-word completion under head ablation."""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch


def load_lambada_examples(num_examples: int = 250, split: str = "test") -> list:
    """Load the LAMBADA (OpenAI variant) evaluation split as raw text
    examples, robustly, including on datasets>=3 where legacy loading
    scripts are unavailable (the parquet conversion is used instead)."""
    from datasets import load_dataset

    last_error = None
    for name in ("lambada_openai", "EleutherAI/lambada_openai"):
        try:
            ds = load_dataset(name)
            key = split if split in ds else list(ds.keys())[0]
            texts = [row["text"] for row in ds[key]]
            return texts[:num_examples]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load LAMBADA: {last_error}")


def split_context_target(text: str) -> Tuple[str, str]:
    parts = text.rsplit(" ", 1)
    if len(parts) != 2:
        return text, ""
    return parts[0], parts[1]


@torch.no_grad()
def evaluate_lambada(
    model,
    tokenizer,
    texts: Sequence[str],
    device: str | torch.device = "cpu",
    topk: int = 1,
    full_word: bool = True,
) -> Dict[str, float]:
    """Top-k acc of first-target-token prediction, full-word greedy accuracy
    and target-word perplexity."""
    model.eval()
    n_correct = 0
    n_full = 0
    target_nll = 0.0
    target_tokens = 0
    scored = 0

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        pairs = [split_context_target(t) for t in chunk]
        ctx_encs = tokenizer([c for c, _ in pairs], add_special_tokens=False)["input_ids"]
        tgt_encs = tokenizer(
            [" " + t if t else "" for _, t in pairs], add_special_tokens=False
        )["input_ids"]
        maxlen = max(len(c) for c in ctx_encs)
        ids = torch.full((len(chunk), maxlen), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        last_pos = torch.zeros(len(chunk), dtype=torch.long)
        for bi, c in enumerate(ctx_encs):
            t = torch.tensor(c, dtype=torch.long)
            ids[bi, : len(c)] = t
            attn[bi, : len(c)] = 1
            last_pos[bi] = len(c) - 1
        ids = ids.to(device)
        attn = attn.to(device)
        out = model(input_ids=ids, attention_mask=attn, use_cache=True)
        # right padding: gather logits at each row's true final context position
        next_logits = out.logits[
            torch.arange(len(chunk), device=out.logits.device), last_pos.to(out.logits.device)
        ]
        top_ids = torch.topk(next_logits, k=topk, dim=-1).indices.cpu().tolist()

        for bi, (ctx, tgt) in enumerate(pairs):
            if not tgt or not tgt_encs[bi]:
                continue
            first_tok = tgt_encs[bi][0]
            if top_ids[bi][0] == first_tok:
                n_correct += 1
            first_tok_val = tgt_encs[bi][-1] if first_tok is None else first_tok
            logprob = torch.log_softmax(next_logits[bi].double(), dim=-1)
            target_nll += float(-logprob[first_tok_val].item())
            target_tokens += 1
            scored += 1
            if full_word and _full_word_match(model, tokenizer, ctx, tgt, device):
                n_full += 1
        del out
    return {
        "top1_accuracy": n_correct / max(1, scored),
        "full_word_accuracy": n_full / max(1, scored),
        "target_word_ppl": math.exp(target_nll / max(1, target_tokens)),
        "num_scored_examples": scored,
        "n_correct": n_correct,
    }


@torch.no_grad()
def _full_word_match(
    model, tokenizer, context: str, target_word: str, device,
    max_new_tokens: int = 12,
) -> bool:
    """Greedy continuation of the context; succeeds when the decoded prefix
    equals the exact target word."""
    out = model(
        tokenizer(context, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device),
        use_cache=True,
    )
    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    tok_ids = [int(next_tok.item())]
    for _ in range(max_new_tokens):
        piece = tokenizer.decode(tok_ids).strip()
        if piece == target_word:
            return True
        first_word = piece.split(" ")[0] if piece else ""
        if first_word and first_word != piece and first_word == target_word:
            return True
        if len(piece) > len(target_word) + 1:
            return False
        if int(next_tok.item()) == tokenizer.eos_token_id:
            return False
        out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        tok_ids.append(int(next_tok.item()))
    return tokenizer.decode(tok_ids).strip() == target_word
