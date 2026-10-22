"""Protect final-only supervision/output and validation quality selection."""
import json
from pathlib import Path
import sys
import pytest
from transformers import AutoTokenizer
from moe_router.gptoss_format import encode_example, final_text, prompt_ids
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from train_gptoss import product_score, match_answer


def test_quality_selection_rejects_single_domain_regression():
    baseline={'gsm8k':.75,'sciq':.75}
    good,allowed=product_score({'gsm8k':.75,'sciq':.75},baseline)
    bad,blocked=product_score({'gsm8k':.70,'sciq':1.0},baseline)
    assert allowed and not blocked and good>bad
    assert product_score({'gsm8k':.72,'sciq':.78},baseline)[1]


def test_private_channels_are_rejected():
    assert final_text('391<|return|>')=='391'
    for raw in ['<|channel|>analysis<|message|>private','text<|start|>assistant','<|message|>private']:
        with pytest.raises(ValueError): final_text(raw)


def test_validation_answer_checks_are_explicit_proxies():
    assert match_answer({'source':'gsm8k','output':'reason\n#### 38'},'Total: 38 plates.')
    assert not match_answer({'source':'gsm8k','output':'#### 38'},'Total: 12 plates.')
    assert match_answer({'source':'sciq','output':'density\nPublished support'},'The quantity is density.')


def test_real_pinned_template_supervises_final_targets_only():
    base=Path('artifacts/internal/native/candidates/gpt-oss-20b/base')
    if not (base/'tokenizer.json').exists(): pytest.skip('Pinned GPT-OSS tokenizer unavailable')
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    with Path('data/native/prepared/v1/train.jsonl').open() as f: row=json.loads(next(f))
    sample,reason=encode_example(row,tokenizer,1024,'auto')
    assert reason is None
    supervised=[t for t,l in zip(sample['input_ids'],sample['labels']) if l!=-100]
    text=tokenizer.decode(supervised,skip_special_tokens=False)
    assert text.startswith('<|channel|>final<|message|>')
    assert row['output'] in text
    assert '<|channel|>analysis' not in text
    ids=prompt_ids(tokenizer,row)
    assert sample['input_ids'][:len(ids)]==ids
    assert all(l==-100 for l in sample['labels'][:len(ids)-len(tokenizer.encode('<|channel|>final<|message|>',add_special_tokens=False))])


def test_stream_never_exposes_channel_headers_across_fragment_boundaries():
    from moe_router.gptoss_format import FinalChannelStream
    for marker in FinalChannelStream.forbidden:
        for cut in range(1,len(marker)):
            guard=FinalChannelStream()
            assert guard.feed('Visible answer'+marker[:cut])=='Visible answer'
            with pytest.raises(ValueError): guard.feed(marker[cut:]+'private content')
    guard=FinalChannelStream()
    assert guard.feed('391<|ret')=='391'
    assert guard.feed('urn|>discard')==''
    assert guard.finish()==''


def test_local_fp32_merge_preserves_frozen_native_experts_and_buffers():
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora.layer import Linear
    from transformers import GptOssConfig, GptOssForCausalLM
    from moe_router.native_export import merge_attention_locally_fp32, runtime_buffer_hashes
    config=GptOssConfig(hidden_size=32,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,num_local_experts=2,
        num_experts_per_tok=1,vocab_size=257,layer_types=['full_attention'])
    config._attn_implementation='eager'
    model=get_peft_model(GptOssForCausalLM(config).to(torch.bfloat16),
        LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for name,value in model.named_parameters():
            if 'lora_B' in name: value.normal_(std=.01)
    frozen={n:v.detach().clone() for n,v in model.get_base_model().state_dict().items() if '.self_attn.' not in n}
    buffers=runtime_buffer_hashes(model.get_base_model())
    observed={};handles=[]
    for name,module in model.named_modules():
        if isinstance(module,Linear):
            def capture(mod,inputs,name=name): observed[name]=inputs[0].detach().float()
            handles.append(module.register_forward_pre_hook(capture))
    with torch.inference_mode(): model(torch.tensor([[2,3,4]]),use_cache=False)
    for h in handles: h.remove()
    merged,checks=merge_attention_locally_fp32(model,observed)
    assert len(checks)==4 and all(c['passed'] for c in checks)
    assert not any(isinstance(m,Linear) for m in merged.modules())
    assert all(torch.equal(value,merged.state_dict()[n]) for n,value in frozen.items())
    assert buffers==runtime_buffer_hashes(merged)
    assert all(p.dtype==torch.bfloat16 for p in merged.parameters())


def test_gateway_blocks_split_private_channel_and_releases_admission(tmp_path):
    import asyncio
    import httpx
    from moe_router.backends.internal_vllm import NativeBackend
    from moe_router.settings import Settings
    base=Path('artifacts/internal/native/candidates/gpt-oss-20b/base')
    if not (base/'tokenizer.json').exists(): pytest.skip('Pinned tokenizer unavailable')
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    key=tmp_path/'engine.token'; key.write_text('e'*48); key.chmod(0o600)
    settings=Settings(tmp_path,tmp_path/'router',tmp_path/'base',api_token='app-secret',
        allow_safety_candidate=True,active_profile='native-internal-v1',native_model_dir=str(base),
        native_engine_token_file=str(key))
    def transport(request):
        data=json.loads(request.content)
        assert data['skip_special_tokens'] is False and len(data['stop_token_ids'])==3
        parts=['Safe answer.','<|chan','nel|>analysisPRIVATE']
        chunks=[{'choices':[{'text':part,'finish_reason':None}]} for part in parts]
        return httpx.Response(200,text=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks))
    async def run():
        backend=NativeBackend(settings,tokenizer,httpx.MockTransport(transport)); backend.load()
        assert backend.final_channel
        event=asyncio.Event(); await backend.acquire('private-check',event)
        events=[e async for e in backend.stream('private-check',[1,2],32,'MATH',event)]
        serialized=json.dumps(events)
        assert 'Safe answer.' in serialized and 'PRIVATE' not in serialized and 'analysis' not in serialized
        assert events[-1][0]=='error' and backend.pending==0 and backend.metrics['failed']==1
    asyncio.run(run())
