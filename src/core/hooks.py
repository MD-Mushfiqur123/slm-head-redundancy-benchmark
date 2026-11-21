"""Non-invasive structural head gating hooks.

``HeadGatingHookManager`` attaches to the output projection of every decoder
layer's multi-head attention module. It

1. captures the (unmasked) concatenated per-head activations entering the
   output projection during forward passes, and
2. captures the gradient of the loss with respect to those activations during
   backward passes, immediately reducing it to per-sample directional scalar
   derivatives ``g_{l,h} = sum_{b,t,dk} dL/dy_{l,h} * y_{l,h}`` so no large
   activation/gradient tensors remain resident.

It additionally applies a binary (or soft) head gate to the concatenated
activations, which yields exact head ablation without touching checkpoint
weights. Architecture detection (``transformer.h`` vs ``model.layers``) is
delegated to :func:`src.core.model_loader.locate_attentions`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .model_loader import AttentionSlicing, locate_attentions


class HeadGatingHookManager:
    """Registers forward / backward hooks on every attention output
    projection and implements exact structural head gating."""

    def __init__(
        self,
        model: nn.Module,
        slicing: Optional[Sequence[AttentionSlicing]] = None,
    ) -> None:
        self.model = model
        self.slicing: List[AttentionSlicing] = (
            list(slicing) if slicing is not None else locate_attentions(model)
        )
        self.num_layers = len(self.slicing)
        self.num_heads = self.slicing[0].num_heads

        self._forward_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._head_outputs: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._per_sample_grads = torch.zeros(
            self.num_layers, self.num_heads, 0, dtype=torch.float64
        )
        self.mask: torch.Tensor = torch.ones(
            self.num_layers, self.num_heads, dtype=torch.float32
        )
        self.masking_enabled: bool = False
        self._install_hooks()

    # ------------------------------------------------------------------ #
    # hook installation
    # ------------------------------------------------------------------ #
    def _install_hooks(self) -> None:
        for li in range(self.num_layers):
            self._forward_handles.append(
                self.slicing[li].o_proj.register_forward_pre_hook(self._make_pre_hook(li))
            )
            self._backward_handles.append(
                self.slicing[li].o_proj.register_full_backward_hook(self._make_bwd_hook(li))
            )

    def _make_pre_hook(self, layer_idx: int):
        def pre_hook(module: nn.Module, args) -> Optional[tuple]:
            x = args[0]
            self._head_outputs[layer_idx] = x.detach()
            if self.masking_enabled and not bool(self.mask[layer_idx].all()):
                x = self._apply_gate(layer_idx, x)
                return (x,) + tuple(args[1:])
            return None

        return pre_hook

    def _make_bwd_hook(self, layer_idx: int):
        def bwd_hook(module: nn.Module, grad_input, grad_output) -> None:
            gy = grad_input[0]
            u = self._head_outputs[layer_idx]
            if gy is None or u is None or gy.shape != u.shape:
                return
            bsz = gy.shape[0]
            per_head = (
                (gy.float() * u.float())
                .view(bsz, gy.shape[1], self.num_heads, -1)
                .sum(dim=(1, 3))
                .transpose(0, 1)
                .double()
            )  # [Nh, B]
            if self._per_sample_grads.shape[2] == 0:
                self._per_sample_grads = torch.zeros(
                    self.num_layers,
                    self.num_heads,
                    bsz,
                    dtype=torch.float64,
                    device="cpu",
                )
            elif self._per_sample_grads.shape[2] < bsz:
                grown = torch.zeros(
                    self.num_layers, self.num_heads, bsz, dtype=torch.float64
                )
                grown[:, :, : self._per_sample_grads.shape[2]] = self._per_sample_grads
                self._per_sample_grads = grown
            self._per_sample_grads[layer_idx] = per_head
            self._head_outputs[layer_idx] = None

        return bwd_hook

    def _apply_gate(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        m = self.mask[layer_idx].to(device=x.device, dtype=x.dtype)
        shaped = x.view(*x.shape[:-1], self.num_heads, self.slicing[layer_idx].d_head)
        shaped = shaped * m.view(1, 1, -1, 1)
        return shaped.reshape(x.shape)

    # ------------------------------------------------------------------ #
    # gating control
    # ------------------------------------------------------------------ #
    def set_mask(self, mask: np.ndarray) -> None:
        """Install a binary head mask [L, Nh]; 0 entries are fully ablated."""
        arr = np.asarray(mask, dtype=np.float32)
        if arr.shape != (self.num_layers, self.num_heads):
            raise ValueError(
                f"mask shape {arr.shape} != {(self.num_layers, self.num_heads)}"
            )
        self.mask = torch.from_numpy(arr)
        self.masking_enabled = True

    def clear_mask(self) -> None:
        self.mask = torch.ones(self.num_layers, self.num_heads)
        self.masking_enabled = False

    # ------------------------------------------------------------------ #
    # gradient access
    # ------------------------------------------------------------------ #
    def get_per_sample_alpha_grads(self) -> torch.Tensor:
        """Return per-sample directional derivatives [L, Nh, B] (float64)."""
        return self._per_sample_grads

    def reset_gradients(self) -> None:
        self._per_sample_grads = torch.zeros(
            self.num_layers, self.num_heads, 0, dtype=torch.float64
        )

    def remove_hooks(self) -> None:
        for h in self._forward_handles + self._backward_handles:
            h.remove()
        self._forward_handles.clear()
        self._backward_handles.clear()
