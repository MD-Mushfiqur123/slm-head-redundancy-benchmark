#!/usr/bin/env python3
"""Inline the generated table bodies into paper/main.tex.

The generated per-column rows produced by ``make_tables.py`` carry the only
copy of the measured numbers; embedding the bodies verbatim keeps
``paper/main.tex`` self-contained and byte-comparable with the CSV
artefacts (4-decimal truth audit) while ``paper/tables/`` remains the
generator output.
"""
from __future__ import annotations

import os
import re

PAPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper")


def main() -> None:
    tex_path = os.path.join(PAPER, "main.tex")
    with open(tex_path) as fh:
        src = fh.read()

    def repl(match: re.Match) -> str:
        stem = match.group(1)
        table_path = os.path.join(PAPER, "tables", f"{stem}.tex")
        with open(table_path) as tfh:
            return tfh.read().rstrip("\n")

    new_src, n = re.subn(r"\\input\{tables/(table\d+)\}", repl, src)
    os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
    with open(tex_path, "w") as fh:
        fh.write(new_src)
    print(f"embedded {n} table bodies")


if __name__ == "__main__":
    main()
