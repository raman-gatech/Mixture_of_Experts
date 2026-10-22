"""Small causal language model for validating internal routing before upcycling."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .layers import SparseMoE


@dataclass
class TinyMoEConfig:
    vocab_size: int = 64
    hidden_size: int = 48
    intermediate_size: int = 96
    num_layers: int = 2
    num_heads: int = 4
    num_experts: int = 3
    top_k: int = 2
    max_length: int = 64
    balance_weight: float = 0.01
    z_weight: float = 0.001

    def __post_init__(self):
        if min(self.vocab_size, self.hidden_size, self.intermediate_size, self.num_layers,
               self.num_heads, self.num_experts, self.max_length) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.hidden_size % self.num_heads or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("Invalid attention heads or expert top_k")


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        dense = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size), nn.GELU(),
                              nn.Linear(config.intermediate_size, config.hidden_size))
        self.moe = SparseMoE.from_dense(dense, config.hidden_size, config.num_experts, config.top_k)

    def forward(self, hidden, mask, token_mask):
        normalized = self.attention_norm(hidden)
        attention, _ = self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)
        hidden = hidden + attention
        ffn, balance, z_loss, counts = self.moe(self.ffn_norm(hidden), token_mask)
        return hidden + ffn, balance, z_loss, counts


class TinyMoELM(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or TinyMoEConfig()
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.position_embeddings = nn.Embedding(self.config.max_length, self.config.hidden_size)
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.num_layers)])
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, token_mask=None):
        if input_ids.ndim != 2 or not input_ids.shape[0]:
            raise ValueError("input_ids must be a nonempty batch of token sequences")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids")
        length = input_ids.shape[1]
        if not 0 < length <= self.config.max_length:
            raise ValueError("Sequence length must be within configured context")
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(torch.arange(length, device=input_ids.device))
        mask = torch.ones(length, length, device=input_ids.device, dtype=torch.bool).triu(1)
        balances, z_losses, counts = [], [], []
        for block in self.blocks:
            hidden, balance, z_loss, dispatched = block(hidden, mask, token_mask)
            balances.append(balance)
            z_losses.append(z_loss)
            counts.append(dispatched)
        logits = self.output(self.norm(hidden))
        balance, z_loss = torch.stack(balances).mean(), torch.stack(z_losses).mean()
        result = dict(logits=logits, balance_loss=balance, z_loss=z_loss, expert_counts=torch.stack(counts))
        if labels is not None:
            if length < 2:
                raise ValueError("Next-token training requires at least two tokens")
            targets = labels[:, 1:].clone()
            if token_mask is not None:
                targets.masked_fill_(~token_mask[:, 1:].bool(), -100)
            if not (targets != -100).any():
                raise ValueError("Next-token loss requires at least one unmasked target")
            lm_loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size),
                                                  targets.reshape(-1), ignore_index=-100)
            result.update(lm_loss=lm_loss, loss=lm_loss + self.config.balance_weight * balance + self.config.z_weight * z_loss)
        return result

    def checkpoint_config(self):
        return asdict(self.config)
