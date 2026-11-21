"""User-supplied corpus loading for calibration and evaluation.

Accepted sources (``--corpus`` in the GPU extension CLI):
  * ``.txt``  : one raw text document per line (blank lines skipped);
  * ``.csv``  : a header row plus a text column (auto-detected, or forced
                with ``text_column``); accepts the largest text-like column
                when several candidates share an equal score;
  * ``.jsonl``: one JSON object per line using the first of the common
                keys ``text``, ``content``, ``body``, ``sentence``;
  * ``hf:<org>/<name>``: any Hugging Face ``datasets`` identifier, loaded
                with the first available split and its first string column.

Rows are returned in the same ``{"text": str}`` shape used by
``calibration_and_eval_sequences`` so a user corpus can replace WikiText-2
for both calibration and evaluation without any pipeline change.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

_TEXT_KEYS = ("text", "content", "body", "sentence", "document")


def _rows_from_txt(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return [{"text": line.strip()} for line in f if line.strip()]


def _rows_from_csv(path: str, text_column: str = "") -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        names = list(csv.DictReader(f).fieldnames or [])
    if not names:
        raise ValueError(f"{path}: empty CSV")
    if text_column:
        if text_column not in names:
            raise ValueError(f"{path}: no column '{text_column}' (found {names})")
        column = text_column
    else:
        totals = {name: 0 for name in names}
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for name in names:
                    totals[name] += len((row.get(name) or "").strip())
        filled = {n: t for n, t in totals.items() if t > 0}
        if not filled:
            raise ValueError(f"{path}: no non-empty columns")
        text_like = [k for k in _TEXT_KEYS if k in filled]
        column = max(text_like, key=lambda k: filled[k]) if text_like \
            else max(filled, key=lambda k: filled[k])
    with open(path, encoding="utf-8", newline="") as f:
        rows = [{"text": (row.get(column) or "").strip()}
                for row in csv.DictReader(f) if (row.get(column) or "").strip()]
    if not rows:
        raise ValueError(f"{path}: no rows with text in column '{column}'")
    return rows


def _rows_from_jsonl(path: str, text_column: str = "") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    key: str = text_column
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not key:
                key = next(k for k in _TEXT_KEYS if k in obj)
            rows.append({"text": str(obj[key]).strip()})
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def load_custom_rows(source: str, text_column: str = "") -> List[Dict[str, str]]:
    """Load a user corpus from a local file or a Hugging Face dataset id.

    ``source`` may be a path ending in ``.txt``/``.csv``/``.jsonl`` or an
    ``hf:org/name`` identifier. Returns ``{"text": str}`` rows, order
    preserved.
    """
    if source.startswith("hf:"):
        from datasets import load_dataset

        repo = source[3:]
        ds = load_dataset(repo)
        split = next(iter(ds)) if isinstance(ds, dict) else "train"
        table = ds[split]
        column = text_column or next(
            (k for k in table.column_names
             if k in _TEXT_KEYS), None) or table.column_names[0]
        rows = [{"text": str(row[column]).strip()} for row in table
                if str(row[column]).strip()]
        if not rows:
            raise ValueError(f"{repo}: no non-empty rows in column '{column}'")
        return rows
    if not os.path.exists(source):
        raise FileNotFoundError(source)
    ext = os.path.splitext(source)[1].lower()
    if ext == ".txt":
        rows = _rows_from_txt(source)
    elif ext in (".csv", ".tsv"):
        rows = _rows_from_csv(source, text_column)
    elif ext == ".jsonl":
        rows = _rows_from_jsonl(source, text_column)
    else:
        raise ValueError(f"{source}: unsupported extension '{ext}' "
                         f"(use .txt, .csv, .tsv or .jsonl, or hf:org/name)")
    if not rows:
        raise ValueError(f"{source}: no text rows found")
    return rows
