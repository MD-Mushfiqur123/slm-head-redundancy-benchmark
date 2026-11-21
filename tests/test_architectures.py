"""Architecture-locator and surgery tests for OPT and GPT-NeoX (Pythia)
layer families, using tiny synthetic models (no downloads required)."""
import unittest

import numpy as np
import torch
import torch.nn as nn

from src.core.model_loader import locate_attentions, locate_layers
from src.core.hooks import HeadGatingHookManager
from src.pruners.global_pruner import GlobalHeadPruner
from src.pruners.structured_pruner import excise_heads


def _toy_opt(nl=2, d=48, nh=6):
    import torch.nn.functional as F

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.dk = d // nh
            self.nh = nh
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, d, bias=False)
            self.v_proj = nn.Linear(d, d, bias=False)
            self.out_proj = nn.Linear(d, d, bias=False)
            self.num_heads = nh
            self.head_dim = self.dk
            self.config = type("C", (), {"num_attention_heads": nh})()

        def forward(self, x):
            q = self.q_proj(x).view(*x.shape[:-1], self.num_heads, self.dk).transpose(1, 2)
            k = self.k_proj(x).view(*x.shape[:-1], self.num_heads, self.dk).transpose(1, 2)
            v = self.v_proj(x).view(*x.shape[:-1], self.num_heads, self.dk).transpose(1, 2)
            a = (q @ k.transpose(-1, -2) * self.dk ** -0.5).softmax(-1)
            out = (a @ v).transpose(1, 2).reshape(*x.shape[:-1], -1)
            return self.out_proj(out)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.fc1 = nn.Linear(d, 4 * d)
            self.fc2 = nn.Linear(4 * d, d)

        def forward(self, x):
            x = x + self.self_attn(x)
            return x + self.fc2(F.relu(self.fc1(x)))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, d)
            self.model = nn.Module()
            self.model.decoder = nn.Module()
            self.model.decoder.layers = nn.ModuleList([Layer() for _ in range(nl)])

            class Conf:
                num_attention_heads = nh

            self.model.decoder.config = Conf
            self.config = Conf
            self.head = nn.Linear(d, 32)

        def forward(self, input_ids, labels=None):
            x = self.emb(input_ids)
            for layer in self.model.decoder.layers:
                x = layer(x)
            logits = self.head(x)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )
            return {"logits": logits, "loss": loss}

    return Model()


def _toy_neox(nl=2, d=48, nh=6):
    import torch.nn.functional as F

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_size = d // nh
            self.nh = nh
            self.query_key_value = nn.Linear(d, 3 * d, bias=False)
            self.dense = nn.Linear(d, d, bias=False)
            self.config = type("C", (), {"num_attention_heads": nh})()

        def forward(self, x):
            dk = self.head_size
            qkv = self.query_key_value(x).view(*x.shape[:-1], -1, 3 * dk).transpose(1, 2)
            q, k, v = qkv.chunk(3, dim=-1)
            a = (q @ k.transpose(-1, -2) * dk ** -0.5).softmax(-1)
            out = (a @ v).transpose(1, 2).reshape(*x.shape[:-1], -1)
            return self.dense(out)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attn()
            self.mlp = nn.Linear(d, d)

        def forward(self, x):
            return x + self.attention(x) + self.mlp(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, d)
            self.gpt_neox = nn.Module()
            self.gpt_neox.layers = nn.ModuleList([Layer() for _ in range(nl)])

            class Conf:
                num_attention_heads = nh

            self.model = nn.Module()
            self.model.config = Conf
            self.config = Conf
            self.head = nn.Linear(d, 32)

        def forward(self, input_ids, labels=None):
            x = self.emb(input_ids)
            for layer in self.gpt_neox.layers:
                x = layer(x)
            logits = self.head(x)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )
            return {"logits": logits, "loss": loss}

    return Model(), nh


def _assert_gated_equals_surgery(model, mask, ratio):
    pruned = excise_heads(model, mask)
    probe = torch.randint(0, 32, (2, 16))
    hm = HeadGatingHookManager(model)
    hm.set_mask(mask)
    with torch.no_grad():
        lg = hm.model(input_ids=probe)["logits"]
        ls = pruned(input_ids=probe)["logits"]
    hm.clear_mask()
    hm.remove_hooks()
    rel = float((lg - ls).abs().max()) / float(ls.abs().max())
    excised = int((torch.as_tensor(mask) == 0).sum())
    return rel, excised


class TestOptTopology(unittest.TestCase):
    def test_locator_finds_opt_decoder_layers(self):
        model = _toy_opt()
        layers = locate_layers(model)
        slicing = locate_attentions(model)
        self.assertEqual(len(layers), 2)
        self.assertEqual(len(slicing), 2)
        self.assertFalse(slicing[0].is_conv1d)
        self.assertEqual(slicing[0].num_heads, 6)
        self.assertEqual(slicing[0].num_kv_heads, 6)

    def test_opt_surgery_equals_gating(self):
        model = _toy_opt()
        nl, nh = 2, 6
        pruner = GlobalHeadPruner(nl, nh, kv_group_size=1)
        imp = np.abs(np.random.RandomState(3).rand(nl, nh))
        mask = pruner.build_mask(imp, 0.34)
        rel, excised = _assert_gated_equals_surgery(model, mask, 0.34)
        self.assertGreater(excised, 0)
        self.assertLess(rel, 1e-6)


class TestNeoXTopology(unittest.TestCase):
    def test_locator_finds_fused_neox_attention(self):
        model, nh = _toy_neox()
        layers = locate_layers(model)
        slicing = locate_attentions(model)
        self.assertEqual(len(layers), 2)
        self.assertTrue(slicing[0].fused_qkv)
        self.assertFalse(slicing[0].is_conv1d)
        self.assertEqual(slicing[0].num_heads, nh)
        self.assertEqual(slicing[0].d_head, 48 // nh)

    def test_neox_surgery_equals_gating(self):
        model, nh = _toy_neox()
        nl = 2
        pruner = GlobalHeadPruner(nl, nh, kv_group_size=1)
        imp = np.abs(np.random.RandomState(5).rand(nl, nh))
        mask = pruner.build_mask(imp, 0.34)
        rel, excised = _assert_gated_equals_surgery(model, mask, 0.34)
        self.assertEqual(excised, int(round(nl * nh * 0.34)))
        self.assertLess(rel, 1e-6)


if __name__ == "__main__":
    unittest.main()
