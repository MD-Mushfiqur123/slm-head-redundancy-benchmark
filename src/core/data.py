"""Dataset / corpus loading with strict calibration-evaluation isolation.

Sources:
  * WikiText-2-raw-v1 ``Salesforce/wikitext`` (validation split).
  * Penn Treebank word-level corpus (canonical `wojzaremba/lstm` release,
    vendored under ``data/ptb`` for reproducibility).
  * LAMBADA OpenAI variant ``EleutherAI/lambada_openai`` (test split).
"""
from __future__ import annotations

import os
import urllib.request
from typing import Dict, List, Sequence, Tuple

import numpy as np

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
PTB_REPO_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data"
LAMBADA_REPO = "EleutherAI/lambada_openai"


def load_wikitext_split(split: str = "validation"):
    from datasets import load_dataset

    return load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG)[split]


def load_ptb_sentences(split: str = "test", cache_dir: str = "data/ptb") -> List[str]:
    """Retrieve the raw Penn Treebank LM corpus (one sentence per line)."""
    local = os.path.join(cache_dir, f"ptb.{split}.txt")
    if not os.path.exists(local):
        os.makedirs(cache_dir, exist_ok=True)
        url = f"{PTB_REPO_URL}/ptb.{split}.txt"
        with urllib.request.urlopen(url) as r:
            data = r.read().decode("utf-8", "replace")
        with open(local, "w", encoding="utf-8") as f:
            f.write(data)
    with open(local, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_lambada_texts(num_examples: int = 250, split: str = "test") -> List[str]:
    from datasets import load_dataset

    ds = load_dataset(LAMBADA_REPO)
    key = split if split in ds else list(ds.keys())[0]
    texts = [row["text"] for row in ds[key]]
    return texts[:num_examples]


def calibration_and_eval_sequences(
    split_rows,
    tokenizer,
    num_calibration: int = 150,
    num_eval: int = 200,
    window: int = 512,
    min_tokens: int = 50,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Cut a corpus into consecutive windows; the first ``num_calibration``
    windows are calibration and the next ``num_eval`` are evaluation, with
    guaranteed zero overlap and evaluation sequences of >= ``min_tokens``."""
    ids: List[int] = []
    for row in split_rows:
        text = row["text"] if isinstance(row, dict) else row
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
    seqs: List[List[int]] = []
    pos = 0
    while pos + min_tokens < len(ids) and len(seqs) < num_calibration + num_eval:
        chunk = ids[pos : pos + window]
        if len(chunk) >= min_tokens:
            seqs.append(chunk)
        pos += window
    calibration = seqs[:num_calibration]
    evaluation = seqs[num_calibration : num_calibration + num_eval]
    return calibration, evaluation


def autocast_batch(entries: Sequence[List[int]], pad_id: int) -> Dict:
    """Batch of token lists -> {input_ids, labels, attention_mask}."""
    import torch

    maxlen = max(len(e) for e in entries)
    input_ids = torch.full((len(entries), maxlen), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(entries), maxlen), dtype=torch.long)
    labels = torch.full((len(entries), maxlen), -100, dtype=torch.long)
    for i, e in enumerate(entries):
        t = torch.tensor(e, dtype=torch.long)
        input_ids[i, : len(e)] = t
        labels[i, : len(e)] = t
        attention_mask[i, : len(e)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
