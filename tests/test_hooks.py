"""Hook mechanics: gradient interception without forward disruption."""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.hooks import HeadGatingHookManager


class Conv1D(torch.nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        torch.nn.init.normal_(w, std=0.02)
        self.weight = torch.nn.Parameter(w)
        self.bias = torch.nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(
            *x.shape[:-1], self.nf
        )


class Attn(torch.nn.Module):
    def __init__(self, d=32, nh=4):
        super().__init__()
        self.embed_dim = d
        self.num_heads = nh
        self.head_dim = d // nh
        self.split_size = d
        self.c_attn = Conv1D(3 * d, d)
        self.c_proj = Conv1D(d, d)

    def forward(self, x):
        q, _, v = self.c_attn(x).split(self.split_size, dim=2)
        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        v = v.view(*v.shape[:-1], self.num_heads, self.head_dim)
        scores = torch.einsum("bnhd,bmhd->bhnm", q, q) / (self.head_dim ** 0.5)
        a = scores.softmax(dim=-1)
        out = torch.einsum("bhnm,bmhd->bnhd", a, v)
        out = out.reshape(*out.shape[:-2], -1)
        return self.c_proj(out)


class ToyLayer(torch.nn.Module):
    def __init__(self, d=32, nh=4):
        super().__init__()
        self.attn = Attn(d, nh)

    def forward(self, x):
        return x + self.attn(x)


class Wrapper(torch.nn.Module):
    """Exposes transformer.h so locate_attentions() resolves the toy stack."""

    def __init__(self, nl=2, d=32, nh=4):
        super().__init__()
        self.emb = torch.nn.Embedding(50, d)
        self.toy_layers_list = torch.nn.ModuleList([ToyLayer(d, nh) for _ in range(nl)])

        class T:
            pass

        self.transformer = T()
        self.transformer.h = self.toy_layers_list
        self.head = torch.nn.Linear(d, 50)

    def forward(self, input_ids, labels=None):
        x = self.emb(input_ids)
        for layer in self.transformer.h:
            x = layer(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        return {"logits": logits, "loss": loss}


class TestHooks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.wrapper = Wrapper(nl=2, d=32, nh=4)
        for p in self.wrapper.parameters():
            p.requires_grad_(False)
        self.wrapper.emb.weight.requires_grad_(True)
        self.hm = HeadGatingHookManager(self.wrapper)

    def test_hooks_capture_gradients_without_disrupting_forward(self):
        ids = torch.randint(0, 50, (2, 7))
        with torch.no_grad():
            base = self.wrapper(input_ids=ids)["logits"]
        out = self.wrapper(input_ids=ids, labels=ids)
        out["loss"].backward()
        grads = self.hm.get_per_sample_alpha_grads()
        self.assertEqual(grads.shape, (2, 4, 2))
        self.assertGreater(float(grads.abs().sum()), 0.0)
        with torch.no_grad():
            again = self.wrapper(input_ids=ids)["logits"]
        self.assertTrue(torch.allclose(base, again))

    def test_gating_changes_output_and_is_reversible(self):
        ids = torch.randint(0, 50, (1, 7))
        with torch.no_grad():
            base = self.wrapper(input_ids=ids)["logits"]
        mask = np.ones((2, 4), dtype=np.float32)
        mask[1, 2] = 0
        self.hm.set_mask(mask)
        with torch.no_grad():
            gated = self.wrapper(input_ids=ids)["logits"]
        self.assertFalse(torch.allclose(base, gated))
        self.hm.clear_mask()
        with torch.no_grad():
            restored = self.wrapper(input_ids=ids)["logits"]
        self.assertTrue(torch.allclose(base, restored))

    def test_grads_shape_per_sample(self):
        ids = torch.randint(0, 50, (3, 7))
        out = self.wrapper(input_ids=ids, labels=ids)
        out["loss"].backward()
        grads = self.hm.get_per_sample_alpha_grads()
        self.assertEqual(grads.shape, (2, 4, 3))
        self.assertEqual(grads.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
