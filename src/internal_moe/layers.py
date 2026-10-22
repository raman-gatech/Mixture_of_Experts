"""Sparse token dispatch with differentiable gates and no token dropping."""

from copy import deepcopy

import torch
from torch import nn


class SparseMoE(nn.Module):
    def __init__(self, hidden_size, experts, top_k=2):
        super().__init__()
        if not experts or not 1 <= top_k <= len(experts):
            raise ValueError("Require experts and 1 <= top_k <= number of experts")
        self.experts = nn.ModuleList(experts)
        self.gate = nn.Linear(hidden_size, len(experts), bias=False)
        self.top_k = top_k

    @classmethod
    def from_dense(cls, dense_ffn, hidden_size, num_experts=3, top_k=2):
        """Clone a non-quantized FFN. Identical experts preserve its initial output."""
        parameters = list(dense_ffn.parameters())
        if not parameters or any(not p.is_floating_point() for p in parameters):
            raise ValueError("Conversion requires a floating-point dense FFN")
        layer = cls(hidden_size, [deepcopy(dense_ffn) for _ in range(num_experts)], top_k)
        return layer.to(device=parameters[0].device, dtype=parameters[0].dtype)

    def forward(self, hidden, token_mask=None):
        """Return hidden states, load-balancing loss, router-z loss, and counts.

        Padding may be excluded from auxiliary losses with token_mask. Expert
        counts record dispatched tokens (including padding), not semantic domains.
        """
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        if not flat.shape[0]:
            raise ValueError("Cannot route an empty token sequence")
        logits = self.gate(flat).float()
        probabilities = logits.softmax(dim=-1)
        scores, indices = probabilities.topk(self.top_k, dim=-1)
        weights = scores / scores.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(flat)
        counts = []
        for expert_id, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == expert_id)
            counts.append(tokens.numel())
            if tokens.numel():
                values = expert(flat.index_select(0, tokens))
                weighted = values * weights[tokens, slots, None].to(values.dtype)
                output = output.index_add(0, tokens, weighted)
        selected = torch.nn.functional.one_hot(indices, len(self.experts)).float().mean(1)
        if token_mask is not None:
            if token_mask.shape != hidden.shape[:-1]:
                raise ValueError("token_mask must match hidden token dimensions")
            mask = token_mask.reshape(-1).to(logits.dtype)
        else:
            mask = logits.new_ones(flat.shape[0])
        denominator = mask.sum().clamp_min(1)
        mean_prob = (probabilities * mask[:, None]).sum(0) / denominator
        mean_selection = (selected * mask[:, None]).sum(0) / denominator
        balance = len(self.experts) * (mean_prob * mean_selection).sum()
        z_loss = (logits.logsumexp(-1).square() * mask).sum() / denominator
        return output.reshape(shape), balance, z_loss, torch.tensor(counts, device=flat.device)
