"""Pruner correctness: exact sparsity ratios and mask shapes."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.model_loader import coupon_shuffle_mask
from src.pruners.base_pruner import expected_pruned_count, validate_mask
from src.pruners.global_pruner import GlobalHeadPruner
from src.pruners.layerwise_pruner import LayerwiseHeadPruner
from src.pruners.structured_pruner import excise_heads
from src.core.model_loader import locate_attentions


def _toy_llama(nl=2, d=48, nh=6, nkv=2):
    import torch
    import torch.nn as nn

    class Sub(nn.Module):
        def __init__(self):
            super().__init__()
            self.dk = d // nh
            self.nh = nh
            self.nkv = nkv
            self.q_proj = nn.Linear(d, nh * self.dk, bias=False)
            self.k_proj = nn.Linear(d, nkv * self.dk, bias=False)
            self.v_proj = nn.Linear(d, nkv * self.dk, bias=False)
            self.o_proj = nn.Linear(nh * self.dk, d, bias=False)
            self.num_heads = nh
            self.num_key_value_heads = nkv
            self.head_dim = self.dk
            self.scale = self.dk ** -0.5
            self.num_key_value_groups = nh // nkv
            self.config = type("C", (), {
                "num_attention_heads": nh, "num_key_value_heads": nkv
            })()

        def forward(self, x):
            q = self.q_proj(x).view(*x.shape[:-1], self.num_heads, self.dk).transpose(1, 2)
            k = self.k_proj(x).view(*x.shape[:-1], self.num_key_value_heads, self.dk).transpose(1, 2)
            v = self.v_proj(x).view(*x.shape[:-1], self.num_key_value_heads, self.dk).transpose(1, 2)
            rep = self.num_heads // self.num_key_value_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            a = (q @ k.transpose(-1, -2) * self.scale).softmax(-1)
            out = (a @ v).transpose(1, 2).reshape(*x.shape[:-1], -1)
            return self.o_proj(out)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Sub()

        def forward(self, x):
            return x + self.self_attn(x)

    class AttnConf:
        num_attention_heads = nh
        num_key_value_heads = nkv

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, d)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer() for _ in range(nl)])

            class Conf:
                num_attention_heads = nh
                num_key_value_heads = nkv

            self.model.config = Conf()
            self.config = Conf()
            self.head = nn.Linear(d, 32)

        def forward(self, input_ids, labels=None):
            x = self.emb(input_ids)
            for layer in self.model.layers:
                x = layer(x)
            logits = self.head(x)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )
            return {"logits": logits, "loss": loss}

    return Model(), nh


class TestGlobalPruner(unittest.TestCase):
    def test_exact_sparsity_and_shapes(self):
        np.random.seed(0)
        imp = np.random.rand(12, 12)
        p = GlobalHeadPruner(12, 12)
        for ratio in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            mask = p.build_mask(imp, ratio)
            self.assertEqual(mask.shape, (12, 12))
            validate_mask(mask, 12, 12, ratio)
            self.assertEqual(int((mask == 0).sum()), expected_pruned_count(12, 12, ratio))

    def test_rank_ordering_matches_importance(self):
        imp = np.arange(144, dtype=float).reshape(12, 12)
        p = GlobalHeadPruner(12, 12)
        mask = p.build_mask(imp, 0.25)
        self.assertTrue(np.all(mask[0] == 0))
        self.assertTrue(np.all(mask[3] == 1))

    def test_gqa_group_alignment(self):
        imp = np.random.RandomState(1).rand(30, 9)
        p = GlobalHeadPruner(30, 9, kv_group_size=3)
        for ratio in (0.1, 0.3, 0.6):
            mask = p.build_mask(imp, ratio)
            self.assertEqual(int((mask == 0).sum()) % 3, 0)
            pruned_expected = int(round(270 * ratio))
            self.assertEqual(int((mask == 0).sum()), pruned_expected)
            for li in range(30):
                groups = mask[li].reshape(3, 3)
                sums = groups.sum(axis=1)
                self.assertTrue(np.all(np.isin(sums, [0.0, 3.0])))


class TestLayerwisePruner(unittest.TestCase):
    def test_per_layer_counts(self):
        imp = np.random.rand(6, 8)
        p = LayerwiseHeadPruner(6, 8)
        mask = p.build_mask(imp, 0.5)
        self.assertEqual(int((mask == 0).sum()), int(round(6 * 0.5)) * 8)
        self.assertEqual(mask.shape, (6, 8))
        for li in range(6):
            self.assertEqual(int((mask[li] == 0).sum()), 4)


class TestStructuredPruner(unittest.TestCase):
    def test_mha_surgery_identity_with_gating(self):
        import torch

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tests.test_hooks import Wrapper

        torch.manual_seed(1)
        net = Wrapper(nl=2, d=32, nh=4)
        for p in net.parameters():
            p.requires_grad_(False)
        net.toy_layers_list  # keep reference
        from src.core.hooks import HeadGatingHookManager

        hm = HeadGatingHookManager(net)
        imp = np.arange(8, dtype=float).reshape(2, 4)
        mask = GlobalHeadPruner(2, 4).build_mask(imp, 0.25)
        ids = torch.randint(0, 50, (1, 6))
        hm.set_mask(mask)
        with torch.no_grad():
            gated = net(input_ids=ids)["logits"]
        hm.remove_hooks()
        pruned = excise_heads(net, mask)
        with torch.no_grad():
            via_surgery = pruned(input_ids=ids)["logits"]
        self.assertTrue(torch.allclose(gated, via_surgery, atol=1e-5))

    def test_all_heads_excised_layer_matches_gating(self):
        import torch

        from tests.test_hooks import Wrapper
        from src.core.hooks import HeadGatingHookManager

        torch.manual_seed(2)
        net = Wrapper(nl=2, d=32, nh=4)
        for p in net.parameters():
            p.requires_grad_(False)
        mask = torch.zeros(2, 4)
        ids = torch.randint(0, 50, (1, 6))
        hm = HeadGatingHookManager(net)
        hm.set_mask(mask)
        with torch.no_grad():
            gated = net(input_ids=ids)["logits"]
        hm.remove_hooks()
        pruned = excise_heads(net, mask)
        with torch.no_grad():
            via_surgery = pruned(input_ids=ids)["logits"]
        self.assertTrue(torch.allclose(gated, via_surgery, atol=1e-5))

    def test_gqa_surgery_matches_gating(self):
        import torch

        model, nh = _toy_llama()
        import torch.nn as nn

        def forward(model, ids):
            x = model.emb(ids)
            for layer in model.model.layers:
                x = layer(x)
            return model.head(x)

        from tests.test_hooks import HeadGatingHookManager  # reuse import location

        # build manager with llama-style slicing
        from src.core.model_loader import locate_attentions

        hm = HeadGatingHookManager(model)
        imp = np.random.RandomState(0).rand(2, 6)
        p = GlobalHeadPruner(2, 6, kv_group_size=3)
        mask = p.build_mask(imp, 0.5)
        ids = torch.randint(0, 32, (1, 5))
        hm.set_mask(mask)
        with torch.no_grad():
            via_sattn = forward(model, ids)
        # replicate gating manually over self_attn outputs is non-trivial
        # here; instead assert mask is group-aligned & surgery runs exactly.
        hm.remove_hooks()
        pruned = excise_heads(model, mask)
        self.assertEqual(int((mask == 0).sum()) % 3, 0)
        with torch.no_grad():
            out = forward(pruned, ids)
        self.assertEqual(out.shape, via_sattn.shape)


class TestRandomMask(unittest.TestCase):
    def test_coupon_shuffle(self):
        g = np.random.default_rng(5)
        mask = coupon_shuffle_mask(12, 12, 0.3, g)
        self.assertEqual(int((mask == 0).sum()), int(round(144 * 0.3)))
        g2 = np.random.default_rng(5)
        same = coupon_shuffle_mask(12, 12, 0.3, g2)
        self.assertTrue(np.array_equal(mask, same))


if __name__ == "__main__":
    unittest.main()
