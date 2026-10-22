"""Real miniature GPT-OSS architecture: native parity, training and standard export."""
import copy

import pytest
import torch
from transformers import GptOssConfig, GptOssForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from moe_router.gptoss_expert_adapter import (
    GptOssExpertAdapter, adapter_state_dict, configure_expert_adapters,
    load_adapter_state_dict, merge_expert_adapters, set_gate_trainable,
)


def tiny_model():
    torch.manual_seed(13)
    config = GptOssConfig(vocab_size=32, hidden_size=16, intermediate_size=12,
                          num_hidden_layers=2, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=8, num_local_experts=4,
                          num_experts_per_tok=2, max_position_embeddings=32,
                          sliding_window=8, pad_token_id=0, eos_token_id=2,
                          attention_dropout=0.0)
    config._attn_implementation = 'eager'
    return GptOssForCausalLM(config)


def inputs():
    return torch.tensor([[3, 5, 8, 9], [4, 2, 6, 7]])


def test_zero_delta_parity_and_freeze_native_base():
    model = tiny_model().eval()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    expected = model(inputs()).logits.detach()
    config = configure_expert_adapters(model, rank=2, alpha=4)
    actual = model(inputs()).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(name.endswith(('_A', '_B')) for name in trainable)
    assert config['layers'] == [0, 1]


def test_gate_warmup_gradients_checkpoint_and_merge_reload(tmp_path):
    model = tiny_model().train()
    configure_expert_adapters(model, rank=2, alpha=3, train_gates=True)
    original_base = {name: p.detach().clone() for name, p in model.named_parameters()
                     if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.02)
    set_gate_trainable(model, False)
    loss = model(inputs(), labels=inputs()).loss
    loss.backward()
    assert all(p.grad is None for name, p in model.named_parameters() if '.router.' in name)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if name.endswith('_B'))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    set_gate_trainable(model, True)
    loss = model(inputs(), labels=inputs()).loss
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if '.router.' in name)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if name.endswith('_A'))
    optimizer.step()
    for name, value in original_base.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
    state = adapter_state_dict(model)
    restored = tiny_model().eval()
    configure_expert_adapters(restored, rank=2, alpha=3)
    load_adapter_state_dict(restored, state)
    expected = model.eval()(inputs()).logits.detach()
    torch.testing.assert_close(restored(inputs()).logits, expected, rtol=0, atol=0)
    result = merge_expert_adapters(restored)
    assert result == {'merged_layers': 2, 'merged_expert_projections': 16}
    assert all(isinstance(layer.mlp.experts, GptOssExperts) for layer in restored.model.layers)
    assert not any(name.endswith(('_A', '_B')) for name in restored.state_dict())
    torch.testing.assert_close(restored(inputs()).logits, expected, rtol=3e-5, atol=2e-6)
    restored.save_pretrained(tmp_path)
    reloaded = GptOssForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(reloaded(inputs()).logits, restored(inputs()).logits, rtol=0, atol=0)


def test_selected_layers_strict_restore_and_invalid_configuration():
    model = tiny_model()
    with pytest.raises(ValueError):
        configure_expert_adapters(model, layer_indices=[2])
    configure_expert_adapters(model, layer_indices=[1])
    assert isinstance(model.model.layers[0].mlp.experts, GptOssExperts)
    state = adapter_state_dict(model)
    assert all('layers.1.' in key for key in state)
    with pytest.raises(ValueError):
        load_adapter_state_dict(model, {})
    invalid = copy.deepcopy(state)
    key = next(iter(invalid))
    invalid[key].flatten()[0] = float('nan')
    with pytest.raises(ValueError):
        load_adapter_state_dict(model, invalid)
    with pytest.raises(ValueError):
        configure_expert_adapters(model)


def test_interleaving_clamping_masked_expert_and_input_gradients():
    model = tiny_model()
    base = model.model.layers[0].mlp.experts
    wrapped = GptOssExpertAdapter(base, 2, 2)
    x = (torch.randn(2, 3, 16) * 100).requires_grad_()
    indices = torch.tensor([[0, 4], [1, 2], [3, 4], [0, 1], [2, 4], [1, 3]])
    scores = torch.tensor([[.8, 0, 0, 0], [0, .4, .6, 0], [0, 0, 0, .7],
                           [.5, .5, 0, 0], [0, 0, 1., 0], [0, .4, 0, .6]])
    torch.testing.assert_close(wrapped(x, indices, scores), base(x, indices, scores), rtol=0, atol=0)
    wrapped(x, indices, scores).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_bf16_base_fp32_adapters_and_native_gate_optimizer():
    model = tiny_model().to(torch.bfloat16).train()
    configure_expert_adapters(model, train_gates=True)
    assert model.model.layers[0].mlp.router.weight.dtype == torch.float32
    assert model.model.layers[0].mlp.experts.gate_up_A.dtype == torch.float32
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model(inputs(), labels=inputs()).loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    optimizer.step()
    merge_expert_adapters(model)
    assert model.model.layers[0].mlp.router.weight.dtype == torch.bfloat16
    with torch.no_grad():
        assert torch.isfinite(model(inputs()).logits).all()


def test_batched_eval_preserves_native_bf16_gemms_and_reduction():
    model = tiny_model().to(torch.bfloat16)
    native = model.model.layers[0].mlp.experts
    wrapped = GptOssExpertAdapter(native, 2, 2).eval()
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    scores = torch.softmax(torch.randn(8, 4, dtype=torch.bfloat16), -1)
    states = x.reshape(-1, 16).repeat(4, 1).view(4, -1, 16)
    gate_up = torch.bmm(states, native.gate_up_proj) + native.gate_up_proj_bias[..., None, :]
    gate = gate_up[..., ::2].clamp(max=native.limit)
    up = gate_up[..., 1::2].clamp(min=-native.limit, max=native.limit)
    activated = (up + 1) * (gate * torch.sigmoid(gate * native.alpha))
    expected = torch.bmm(activated, native.down_proj) + native.down_proj_bias[..., None, :]
    expected = expected.view(4, 2, -1, 16) * scores.transpose(0, 1).view(4, 2, -1)[..., None]
    torch.testing.assert_close(wrapped._forward_batched(x, scores), expected.sum(0), rtol=0, atol=0)
    with torch.no_grad():
        wrapped.gate_up_B.normal_(std=.03)
        wrapped.down_B.normal_(std=.03)
    actual = wrapped._forward_batched(x, scores)
    assert torch.isfinite(actual).all()
    assert not torch.equal(actual, expected.sum(0))
