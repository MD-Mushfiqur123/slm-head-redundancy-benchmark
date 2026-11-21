"""Eval engines: sliding-window PPL consistency and LAMBADA utilities."""
import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.perplexity import build_windows, evaluate_sliding_perplexity
from src.eval.downstream import split_context_target


class ToyLM(torch.nn.Module):
    """Fixed softmax model: P(next token) uniform -> known exact NLL."""

    def __init__(self, vocab=100, d=16):
        super().__init__()
        self.vocab = vocab
        self.emb = torch.nn.Embedding(vocab, d)

    def forward(self, input_ids, labels=None, use_cache=False, attention_mask=None):
        logits = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], self.vocab),
            dtype=torch.float32,
        )
        out = {"logits": logits}
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab), labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out


class TestSlidingWindowPPL(unittest.TestCase):
    def test_uniform_model_gives_expected_ppl(self):
        model = ToyLM(vocab=100)
        pad_id = 0
        seqs = [list(np.random.RandomState(0).randint(0, 100, size=(300,)))]
        res = evaluate_sliding_perplexity(
            model, seqs, pad_id=pad_id, batch_size=2, stride=256, max_length=512
        )
        self.assertAlmostEqual(res["perplexity"], 100.0, places=1)
        self.assertEqual(res["total_tokens"], sum(1 for s in seqs for _ in s) - len(seqs))

    def test_ppl_matches_manual_computation(self):
        class biased(ToyLM):
            def forward(self, input_ids, labels=None, use_cache=False, attention_mask=None):
                logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 10))
                logits[..., 0] = 2.0  # bias towards token 0
                out = {"logits": logits}
                if labels is not None:
                    out["loss"] = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, 10), labels.reshape(-1), ignore_index=-100
                    )
                return out

        model = biased()
        seqs = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 3]
        res = evaluate_sliding_perplexity(model, seqs, pad_id=0, batch_size=1, stride=256, max_length=512)
        logprobs = torch.log_softmax(torch.tensor([2.0] + [0.0] * 9).double(), dim=-1)
        # token 0 scored 2x, tokens 1..9 scored 3x each; 29 scored tokens
        expected = math.exp(-(logprobs[0] * 2 + sum(logprobs[1:]) * 3) / 29)
        self.assertAlmostEqual(res["perplexity"], float(expected), places=5,
                               msg="engine and manual NLL must agree to fp accumulation order")

    def test_window_construction(self):
        seq = list(range(1200))
        windows = build_windows(seq, max_length=512, stride=256)
        # coverage: union of windows covers the whole sequence
        covered = set()
        for w in windows:
            covered.update(w["tokens"])
        self.assertEqual(covered, set(seq))
        # no token scored twice within a sequence
        scored = [len(w["tokens"]) - w["score_from"] for w in windows]
        self.assertEqual(sum(scored), len(seq) - 1)

    def test_exact_tokenize_invariance_on_batching(self):
        model = ToyLM(vocab=50)
        seqs = [list(range(30)), list(range(30))]
        r1 = evaluate_sliding_perplexity(model, seqs, pad_id=0, batch_size=1)
        r2 = evaluate_sliding_perplexity(model, seqs, pad_id=0, batch_size=2)
        self.assertAlmostEqual(r1["perplexity"], r2["perplexity"], places=9)


class TestLambadaUtils(unittest.TestCase):
    def test_split_context_target(self):
        ctx, tgt = split_context_target("the quick brown fox jumps over the lazy dog")
        self.assertEqual(ctx, "the quick brown fox jumps over the lazy")
        self.assertEqual(tgt, "dog")

    def test_no_space_fallback(self):
        ctx, tgt = split_context_target("singleword")
        self.assertEqual(tgt, "")


if __name__ == "__main__":
    unittest.main()
