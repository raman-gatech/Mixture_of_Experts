"""Export gates detect numeric divergence and verify native cached decoding."""
import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import GptOssConfig, GptOssForCausalLM

spec = importlib.util.spec_from_file_location('expert_export', Path(__file__).resolve().parents[1] / 'scripts/qualify_gptoss_expert_export.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_distribution_and_standard_checkpoint_gates():
    reference = torch.tensor([[2., 1., -1.]])
    assert export.distribution_check(reference, reference)['passed']
    assert not export.distribution_check(reference, -reference)['passed']
    assert not export.distribution_check(reference, torch.full_like(reference, float('nan')))['passed']
    export.assert_standard_keys(['model.layers.0.mlp.experts.gate_up_proj', 'model.layers.0.mlp.router.weight'])
    with pytest.raises(ValueError):
        export.assert_standard_keys(['model.layers.0.mlp.experts.gate_up_A'])
    assert export.mutable_tensor('model.layers.0.mlp.experts.gate_up_proj')
    assert not export.mutable_tensor('model.layers.0.mlp.experts.gate_up_proj_bias')
    assert not export.mutable_tensor('model.layers.0.self_attn.q_proj.weight')


def test_calibration_is_explicit_nonmutating_and_scales_gates_from_parent():
    state = {'x.gate_up_A': torch.tensor([2.]), 'x.gate_up_B': torch.tensor([4.]),
             'x.router.weight': torch.tensor([.4])}
    parent = {'x.router.weight': torch.tensor([.2])}
    half = export.calibrated_state(state, parent, .5)
    assert torch.equal(half['x.gate_up_A'], state['x.gate_up_A'])
    assert torch.equal(half['x.gate_up_B'], torch.tensor([2.]))
    assert torch.allclose(half['x.router.weight'], torch.tensor([.3]))
    assert state['x.gate_up_B'].item() == 4.
    assert all(torch.equal(export.calibrated_state(state,parent,1.)[n],v) for n,v in state.items())
    with pytest.raises(ValueError):
        export.calibrated_state(state,parent,.75)


def test_probe_cache_and_generation_repeat_exactly_on_real_architecture():
    torch.manual_seed(7)
    config = GptOssConfig(vocab_size=32, hidden_size=16, intermediate_size=12,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_local_experts=4, num_experts_per_tok=2,
        max_position_embeddings=32, sliding_window=8, pad_token_id=0, eos_token_id=2,
        attention_dropout=0.0)
    config._attn_implementation = 'eager'
    model = GptOssForCausalLM(config).eval()
    class Tokenizer:
        eos_token_id = 2
        def decode(self, ids, **kwargs):
            return ','.join(str(int(value)) for value in ids)
    ids = torch.tensor([[3, 4, 5, 6]])
    first = export.probe(model, ids, Tokenizer(), 3)
    second = export.probe(model, ids, Tokenizer(), 3)
    assert first['cache']['passed']
    assert torch.equal(first['logits'], second['logits'])
    assert torch.equal(first['cached_logits'], second['cached_logits'])
    assert torch.equal(first['generation'], second['generation'])
    assert export.tensor_hash(model.model.layers[0].mlp.router.weight) == export.tensor_hash(model.model.layers[0].mlp.router.weight.clone())
