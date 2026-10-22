"""A calibrated adapter retains a real learned update and correct projection algebra."""
from pathlib import Path
import sys
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from calibrate_gptoss_adapter import scale_state


def test_shrinkage_preserves_attention_update_algebra_and_frozen_weights(tmp_path):
    from safetensors.torch import load_file
    model = get_peft_model(GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=8, n_head=1, n_layer=1)),
        LoraConfig(r=2, lora_alpha=4, target_modules=['c_attn'], task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, value in model.named_parameters():
            if 'lora_B.' in name: value.normal_(std=.1)
    model.save_pretrained(tmp_path)
    state = load_file(tmp_path / 'adapter_model.safetensors')
    original = {key: value.clone() for key, value in state.items()}
    scaled = scale_state(state, .25)
    a = next(value for key, value in state.items() if 'lora_A.' in key)
    b = next(value for key, value in state.items() if 'lora_B.' in key)
    sa = next(value for key, value in scaled.items() if 'lora_A.' in key)
    sb = next(value for key, value in scaled.items() if 'lora_B.' in key)
    assert torch.equal(a, sa)
    assert torch.allclose(sb @ sa, .25 * (b @ a))
    assert torch.count_nonzero(sb) > 0
    assert all(torch.equal(value, original[key]) for key, value in state.items())
    for invalid in (0, 1e-8, 1, -1):
        with pytest.raises(ValueError): scale_state(state, invalid)
    with pytest.raises(ValueError): scale_state({'lm_head.weight': torch.ones(2,2)}, .25)
