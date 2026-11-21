"""Offline tests for user-supplied corpus loading (no network)."""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.custom_dataset import load_custom_rows  # noqa: E402


class TestCustomDataset(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp()

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_txt_lines(self):
        p = self._write("c.txt", "alpha beta gamma\n\n   \ndelta epsilon\n")
        rows = load_custom_rows(p)
        self.assertEqual(rows, [{"text": "alpha beta gamma"},
                                {"text": "delta epsilon"}])

    def test_csv_auto_column(self):
        p = self._write("c.csv", "id,text\n1,first sentence\n2,second sentence\n")
        rows = load_custom_rows(p)
        self.assertEqual([r["text"] for r in rows],
                         ["first sentence", "second sentence"])

    def test_csv_forced_column(self):
        p = self._write("c.csv", "a,b\nxx,one text\nyy,two text\n")
        rows = load_custom_rows(p, text_column="b")
        self.assertEqual([r["text"] for r in rows], ["one text", "two text"])

    def test_csv_missing_column(self):
        p = self._write("c.csv", "a\n1\n")
        with self.assertRaises(ValueError):
            load_custom_rows(p, text_column="zzz")

    def test_jsonl_first_key(self):
        p = self._write("c.jsonl",
                        json.dumps({"content": "one two"}) + "\n"
                        + json.dumps({"content": "three four"}) + "\n")
        rows = load_custom_rows(p)
        self.assertEqual([r["text"] for r in rows], ["one two", "three four"])

    def test_jsonl_forced_key(self):
        p = self._write("c.jsonl",
                        json.dumps({"text": "a", "content": "b"}) + "\n")
        rows = load_custom_rows(p, text_column="content")
        self.assertEqual(rows[0]["text"], "b")

    def test_unsupported_extension(self):
        p = self._write("c.bin", "\x00\x01")
        with self.assertRaises(ValueError):
            load_custom_rows(p)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_custom_rows(os.path.join(self.dir, "nope.txt"))

    def test_empty_rows_rejected(self):
        p = self._write("c.txt", "\n\n")
        with self.assertRaises(ValueError):
            load_custom_rows(p)


if __name__ == "__main__":
    unittest.main()
