"""Training-only low-rank updates to GPT-OSS's actual packed FFN experts.

Forward computes only selected expert/token products. It never constructs a dense
[E, in, out] delta. Merge restores the original Transformers expert modules so a
saved model needs no custom code. Native expert biases and all other base weights
remain frozen; optional router training changes the existing native gate.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class GptOssExpertAdapter(nn.Module):
    """Low-rank gate/up and down updates preserving native interleaved SwiGLU."""

    def __init__(self, experts: nn.Module, rank: int, alpha: float):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("rank and alpha must be positive and finite")
        if experts.gate_up_proj.ndim != 3 or experts.down_proj.ndim != 3:
            raise ValueError("Expected dequantized GPT-OSS packed expert parameters")
        self.rank, self.alpha, self.scaling = rank, alpha, alpha / rank
        self.num_experts, self.hidden_size, twice_intermediate = experts.gate_up_proj.shape
        self.intermediate_size = twice_intermediate // 2
        if twice_intermediate % 2 or tuple(experts.down_proj.shape) != (
            self.num_experts, self.intermediate_size, self.hidden_size
        ):
            raise ValueError("Incompatible GPT-OSS expert projection layout")
        self.alpha_activation, self.limit = experts.alpha, experts.limit
        # Preserve exact parameter objects/names and retain the native module for merge.
        object.__setattr__(self, "_native_experts", experts)
        for name in ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"):
            parameter = getattr(experts, name)
            parameter.requires_grad_(False)
            self.register_parameter(name, parameter)
        device = self.gate_up_proj.device
        # FP32 master factors prevent small low-rank updates vanishing in BF16 storage.
        self.gate_up_A = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, rank, device=device, dtype=torch.float32))
        self.gate_up_B = nn.Parameter(torch.zeros(self.num_experts, rank, twice_intermediate, device=device, dtype=torch.float32))
        self.down_A = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, rank, device=device, dtype=torch.float32))
        self.down_B = nn.Parameter(torch.zeros(self.num_experts, rank, self.hidden_size, device=device, dtype=torch.float32))
        nn.init.normal_(self.gate_up_A, std=1 / math.sqrt(self.hidden_size))
        nn.init.normal_(self.down_A, std=1 / math.sqrt(self.intermediate_size))
        self.train(experts.training)

    def _update(self, inputs: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Explicit casts also support BF16 inference outside an autocast context.
        return ((inputs @ a.to(inputs.dtype)) @ b.to(inputs.dtype)) * self.scaling

    def _forward_batched(self, hidden_states, routing_weights):
        """Preserve the native CUDA evaluation GEMMs and reduction order.

        Selected-token training and native batched evaluation round differently
        in BF16. Evaluation must use the same batched path as the parent rather
        than silently changing its unadapted predictions. Low-rank products are
        computed on activations, never as a dense packed weight delta.
        """
        batch_size = hidden_states.shape[0]
        states = hidden_states.reshape(-1, self.hidden_size)
        states = states.repeat(self.num_experts, 1).view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
        update = torch.bmm(torch.bmm(states, self.gate_up_A.to(states.dtype)),
                           self.gate_up_B.to(states.dtype)) * self.scaling
        gate_up = gate_up + update
        gate = gate_up[..., ::2].clamp(max=self.limit)
        up = gate_up[..., 1::2].clamp(min=-self.limit, max=self.limit)
        activated = (up + 1) * (gate * torch.sigmoid(gate * self.alpha_activation))
        result = torch.bmm(activated, self.down_proj) + self.down_proj_bias[..., None, :]
        result = result + torch.bmm(torch.bmm(activated, self.down_A.to(activated.dtype)),
                                    self.down_B.to(activated.dtype)) * self.scaling
        result = result.view(self.num_experts, batch_size, -1, self.hidden_size)
        result = result * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
        return result.sum(dim=0)

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        if router_indices is None or routing_weights is None:
            raise ValueError("Native router indices and scores are required")
        if hidden_states.device.type != 'cpu' and not self.training:
            return self._forward_batched(hidden_states, routing_weights)
        original_shape = hidden_states.shape
        states = hidden_states.reshape(-1, self.hidden_size)
        output = torch.zeros_like(states)
        # Non-differentiable top-k selection; routing weights retain their graph.
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(router_indices, self.num_experts + 1).permute(2, 1, 0)
            active = mask.sum(dim=(-1, -2)).nonzero().flatten()
        for expert in active:
            if expert == self.num_experts:  # native masking sentinel
                continue
            with torch.no_grad():
                _, token_indices = torch.where(mask[expert])
            selected = states[token_indices]
            gate_up = selected @ self.gate_up_proj[expert] + self.gate_up_proj_bias[expert]
            gate_up = gate_up + self._update(selected, self.gate_up_A[expert], self.gate_up_B[expert])
            gate = gate_up[..., ::2].clamp(max=self.limit)
            up = gate_up[..., 1::2].clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha_activation)
            activated = (up + 1) * glu
            result = activated @ self.down_proj[expert] + self.down_proj_bias[expert]
            result = result + self._update(activated, self.down_A[expert], self.down_B[expert])
            weighted = result * routing_weights[token_indices, expert, None]
            output.index_add_(0, token_indices, weighted.to(states.dtype))
        return output.view(original_shape)


def _mlps(model):
    return [(name, module) for name, module in model.named_modules()
            if hasattr(module, "experts") and hasattr(module, "router")
            and hasattr(module.router, "top_k")]


def configure_expert_adapters(model, rank: int = 2, alpha: float = 2.0,
                              layer_indices: Iterable[int] | None = None,
                              train_gates: bool = False):
    """Freeze the base, adapt chosen layers, and optionally enable native gates."""
    mlps = _mlps(model)
    if not mlps or any(isinstance(mlp.experts, GptOssExpertAdapter) for _, mlp in mlps):
        raise ValueError("Expected an unwrapped native GPT-OSS model")
    selected = set(range(len(mlps))) if layer_indices is None else set(layer_indices)
    if not selected or not selected.issubset(range(len(mlps))):
        raise ValueError("layer_indices must select existing native layers")
    if not isinstance(rank, int) or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("rank and alpha must be positive and finite")
    model.requires_grad_(False)
    for index, (_, mlp) in enumerate(mlps):
        if index in selected:
            mlp.experts = GptOssExpertAdapter(mlp.experts, rank, alpha)
    set_gate_trainable(model, train_gates)
    return {"rank": rank, "alpha": alpha, "layers": sorted(selected),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def set_gate_trainable(model, enabled: bool, *, fp32: bool = True):
    """Enable gates in adapted layers only; warmup can keep these frozen.

    FP32 gates require autocast around a BF16 model's forward (including evaluation).
    This keeps optimizer updates in FP32 without changing the standard router class.
    """
    count = 0
    for _, mlp in _mlps(model):
        if not isinstance(mlp.experts, GptOssExpertAdapter):
            continue
        if fp32:
            mlp.router.float()
        mlp.router.requires_grad_(enabled)
        count += sum(parameter.numel() for parameter in mlp.router.parameters())
    return count


def _adapter_parameters(model):
    for name, mlp in _mlps(model):
        if isinstance(mlp.experts, GptOssExpertAdapter):
            for suffix in ("gate_up_A", "gate_up_B", "down_A", "down_B"):
                yield f"{name}.experts.{suffix}", getattr(mlp.experts, suffix)
            for suffix, parameter in mlp.router.named_parameters():
                yield f"{name}.router.{suffix}", parameter


def adapter_state_dict(model):
    """Return CPU copies of all factors and selected gates, including warmup gates."""
    return {name: parameter.detach().cpu().clone() for name, parameter in _adapter_parameters(model)}


def load_adapter_state_dict(model, state):
    """Strictly restore factor/gate tensors into an identically configured model."""
    expected = dict(_adapter_parameters(model))
    if not expected or set(state) != set(expected):
        raise ValueError("Adapter checkpoint keys do not match configured layers")
    for name, parameter in expected.items():
        value = state[name]
        if value.shape != parameter.shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid adapter tensor: {name}")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


@torch.no_grad()
def merge_expert_adapters(model):
    """Merge one expert projection at a time and restore standard native modules.

    FP32 factor products are rounded once to the base dtype. BF16 output parity
    needs numerical/output qualification; mathematical equivalence is not bitwise.
    """
    mlps = [(name, mlp) for name, mlp in _mlps(model)
            if isinstance(mlp.experts, GptOssExpertAdapter)]
    if not mlps:
        raise ValueError("No expert adapters to merge")
    if any(not torch.isfinite(value).all() for _, value in _adapter_parameters(model)):
        raise ValueError("Cannot merge non-finite factors or gates")
    count = 0
    for _, mlp in mlps:
        wrapped = mlp.experts
        for expert in range(wrapped.num_experts):
            for prefix in ("gate_up", "down"):
                parameter = getattr(wrapped, prefix + "_proj")[expert]
                a = getattr(wrapped, prefix + "_A")[expert].float()
                b = getattr(wrapped, prefix + "_B")[expert].float()
                merged = parameter.float().addmm(a, b, alpha=wrapped.scaling)
                if not torch.isfinite(merged).all():
                    raise ValueError(f"Non-finite merged {prefix} projection for expert {expert}")
                parameter.copy_(merged.to(parameter.dtype))
                count += 1
        native = wrapped._native_experts
        native.train(wrapped.training)
        mlp.experts = native
        mlp.router.to(dtype=native.gate_up_proj.dtype)
        mlp.router.requires_grad_(False)
    return {"merged_layers": len(mlps), "merged_expert_projections": count}
