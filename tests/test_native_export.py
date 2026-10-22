"""Regression for the real OLMoE buffer mismatch invisible to weight hashes."""
import copy
import pytest
import torch
pytest.importorskip('transformers')
from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding
from moe_router.native_export import cast_export_weights, runtime_buffer_hashes


def test_export_preserves_reconstructed_rotary_frequencies_and_outputs():
    model=torch.nn.Module()
    model.register_parameter('weight',torch.nn.Parameter(torch.ones(2)))
    model.rotary_emb=OlmoeRotaryEmbedding(OlmoeConfig(hidden_size=64,num_attention_heads=4))
    unsafe=copy.deepcopy(model).to(dtype=torch.bfloat16)
    corrected=cast_export_weights(copy.deepcopy(model),torch.bfloat16)
    # Serialized weights alone cannot reveal this discrepancy.
    assert list(unsafe.state_dict())==['weight']
    assert torch.equal(unsafe.state_dict()['weight'],corrected.state_dict()['weight'])
    fresh=copy.deepcopy(model)
    assert runtime_buffer_hashes(corrected)==runtime_buffer_hashes(fresh)
    assert runtime_buffer_hashes(unsafe)!=runtime_buffer_hashes(fresh)
    assert corrected.weight.dtype==torch.bfloat16
    positions=torch.arange(128).unsqueeze(0)
    hidden=torch.zeros(1,128,64,dtype=torch.bfloat16)
    expected=fresh.rotary_emb(hidden,positions)
    actual=corrected.rotary_emb(hidden,positions)
    rounded=unsafe.rotary_emb(hidden,positions)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    assert any(not torch.equal(a,b) for a,b in zip(expected,rounded))
