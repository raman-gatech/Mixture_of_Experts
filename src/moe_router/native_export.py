"""Keep runtime-only positional buffers canonical when changing weight dtype."""
import hashlib
import torch


def runtime_buffers(model):
    for prefix,module in model.named_modules():
        for name in sorted(module._non_persistent_buffers_set):
            value=module._buffers.get(name)
            if value is not None:
                yield '.'.join(filter(None,[prefix,name])),module,name,value


def cast_export_weights(model,dtype):
    # Module.to() also casts nonpersistent buffers. OLMoE reconstructs inv_freq
    # in FP32, so rounding it here changes behavior without changing state_dict.
    saved=[(module,name,value) for _,module,name,value in runtime_buffers(model)]
    model.to(dtype=dtype)
    for module,name,value in saved:
        module.register_buffer(name,value,persistent=False)
    return model


def runtime_buffer_hashes(model):
    return {key:dict(dtype=str(value.dtype),shape=list(value.shape),
                    sha256=hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
            for key,_,_,value in runtime_buffers(model)}


def merge_attention_locally_fp32(model, observed_inputs):
    """Check each trained attention projection in FP32 without casting a 21B base."""
    from peft.tuners.lora.layer import Linear
    checks = []
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, Linear)]
    if not layers or set(observed_inputs) != {name for name, _ in layers}:
        raise ValueError('Every adapted projection needs observed real-text activations')
    with torch.inference_mode():
        for name, module in layers:
            if '.self_attn.' not in name or module.merged:
                raise ValueError('Expected fresh unmerged attention-only LoRA')
            if module.base_layer.weight.dtype != torch.bfloat16:
                raise ValueError('Expected qualified BF16 base weights')
            if any(True for _ in runtime_buffers(module)):
                raise ValueError('Projection unexpectedly contains runtime buffers')
            # Only this small projection is FP32; sparse FFN tensors stay BF16.
            module.float()
            inputs = observed_inputs[name].to(device=module.base_layer.weight.device, dtype=torch.float32)
            before = module(inputs).float()
            module.merge(safe_merge=True)
            after = module(inputs).float()
            check = dict(name=name, max_abs=float((before-after).abs().max()),
                passed=bool(torch.allclose(before, after, atol=.001, rtol=.0001)))
            checks.append(check)
            module.to(dtype=torch.bfloat16)
            if not check['passed']:
                raise RuntimeError('Projection FP32 merge failed: '+name)
    return model.unload().eval(), checks
