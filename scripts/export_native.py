"""Verify LoRA merging in FP32, then verify the BF16 export against itself on reload."""
import argparse
import gc
import json
import os
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_router.native_data import system_for
from moe_router.artifacts import sha256, inspect_safetensors
from moe_router.native_export import cast_export_weights, runtime_buffer_hashes


def evaluate(model, ids, tokenizer):
    with torch.inference_mode():
        logits=model(ids,use_cache=False).logits[:,-1].float().cpu()
        output=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=32,
                              do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id).cpu()
    return logits,output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():p.error('Requires allocated CUDA GPU')
    status=json.loads((args.run/'training-status.json').read_text())
    provenance=json.loads((args.run/'provenance.json').read_text())
    if not status['completed'] or status['profile_only'] or provenance['profile_only']:
        raise ValueError('Only completed substantial SFT can be exported')
    if args.output.exists():raise FileExistsError(args.output)
    base=Path(provenance['identity']['config']['base'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    # Probe the allocated device before shared-filesystem model loading. Preserve
    # resource evidence without enumerating or interfering with other GPUs.
    free,total=torch.cuda.mem_get_info(0)
    device=dict(node=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],
                name=torch.cuda.get_device_name(0),free_bytes_before_load=free,
                total_bytes=total,torch_version=torch.__version__)
    print(json.dumps(dict(export_device=device)),flush=True)
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer=AutoTokenizer.from_pretrained(args.run/'adapter',local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.bfloat16,
                                             device_map={'':0},attn_implementation='sdpa')
    model=PeftModel.from_pretrained(model,args.run/'adapter').eval()
    nonzero=sum(int(torch.count_nonzero(v).item()) for n,v in model.named_parameters() if 'lora_B' in n)
    if not nonzero:raise ValueError('No learned nonzero LoRA B values')
    probes=[]
    for mode,text in [('MATH','What is 23 times 17?'),('SCIENCE','Why does ice float?'),
                      ('GENERAL','Give three tips for studying.'),('SAFETY','I feel overwhelmed by school.')]:
        ids=tokenizer.apply_chat_template([{'role':'system','content':system_for(mode)},
            {'role':'user','content':text}],tokenize=True,add_generation_prompt=True,return_tensors='pt').cuda()
        before_bf16,generation_bf16=evaluate(model,ids,tokenizer)
        probes.append(dict(mode=mode,ids=ids.cpu(),unmerged_bf16=before_bf16,
                           unmerged_bf16_generation=generation_bf16))
    # FP32 separates algebraic merge correctness from BF16 rounding/routing changes.
    model.float()
    for probe in probes:
        logits,output=evaluate(model,probe['ids'].cuda(),tokenizer)
        probe.update(unmerged_fp32=logits,unmerged_fp32_generation=output)
    merged=model.merge_and_unload(safe_merge=True).eval()
    merge_checks=[]
    for probe in probes:
        logits,output=evaluate(merged,probe['ids'].cuda(),tokenizer)
        before=probe['unmerged_fp32']
        merge_checks.append(dict(mode=probe['mode'],precision='float32',max_abs=float((before-logits).abs().max()),
            mean_abs=float((before-logits).abs().mean()),argmax_equal=bool(torch.equal(before.argmax(-1),logits.argmax(-1))),
            generation_equal=bool(torch.equal(probe['unmerged_fp32_generation'],output)),
            passed=bool(torch.allclose(before,logits,atol=.001,rtol=.0001))))
    cast_export_weights(merged,torch.bfloat16)
    expected_runtime_buffers=runtime_buffer_hashes(merged)
    if expected_runtime_buffers.get('model.rotary_emb.inv_freq',{}).get('dtype')!='torch.float32':
        raise ValueError('Canonical OLMoE rotary frequencies must remain FP32')
    cast_checks=[]
    for probe in probes:
        logits,output=evaluate(merged,probe['ids'].cuda(),tokenizer)
        probe.update(merged_bf16=logits,merged_bf16_generation=output)
        before=probe['unmerged_bf16']
        kl=torch.nn.functional.kl_div(logits.log_softmax(-1),before.softmax(-1),reduction='batchmean')
        cast_checks.append(dict(mode=probe['mode'],max_abs=float((before-logits).abs().max()),
            kl_unmerged_to_merged=float(kl),generation_equal=bool(torch.equal(probe['unmerged_bf16_generation'],output)),
            unmerged_answer=tokenizer.decode(probe['unmerged_bf16_generation'][0,probe['ids'].shape[1]:],skip_special_tokens=True),
            merged_answer=tokenizer.decode(output[0,probe['ids'].shape[1]:],skip_special_tokens=True),
            review_status='unassessed; rounding differences are not silently treated as answer-quality acceptance'))
    args.output.mkdir(parents=True,mode=0o700)
    merged.config.use_cache=True
    merged.generation_config.use_cache=True
    merged.save_pretrained(args.output,safe_serialization=True,max_shard_size='4GB')
    tokenizer.save_pretrained(args.output)
    # Full tensor hashes prove serialization, including frozen gates and experts.
    def tensor_hashes(instance):
        import hashlib
        return {name:hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                for name,value in instance.state_dict().items()}
    expected_tensors=tensor_hashes(merged)
    del model,merged
    gc.collect();torch.cuda.empty_cache()
    reloaded=AutoModelForCausalLM.from_pretrained(args.output,local_files_only=True,torch_dtype=torch.bfloat16,
                                                device_map={'':0},attn_implementation='sdpa').eval()
    tensor_equal=expected_tensors==tensor_hashes(reloaded)
    actual_runtime_buffers=runtime_buffer_hashes(reloaded)
    runtime_equal=expected_runtime_buffers==actual_runtime_buffers
    reload_checks=[]
    for probe in probes:
        logits,output=evaluate(reloaded,probe['ids'].cuda(),tokenizer)
        reload_checks.append(dict(mode=probe['mode'],max_abs=float((probe['merged_bf16']-logits).abs().max()),
            logits_equal=bool(torch.equal(probe['merged_bf16'],logits)),
            generation_equal=bool(torch.equal(probe['merged_bf16_generation'],output))))
    report=dict(merge=merge_checks,reload=reload_checks,bf16_rounding_answer_review=cast_checks,
        all_exported_tensors_equal=tensor_equal,tensors_verified=len(expected_tensors),
        runtime_buffers_before_save=expected_runtime_buffers,runtime_buffers_after_reload=actual_runtime_buffers,
        all_runtime_buffers_equal=runtime_equal,
        device=device,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
        numeric_tolerance={'fp32_merge_atol':.001,'fp32_merge_rtol':.0001,'bf16_reload':'exact tensors/logits/generation'},
        rationale='Test algebraic merging in FP32; compare persisted BF16 model with the same pre-save BF16 model. Record unmerged-vs-merged BF16 behavior separately for actual answer review.',
        learned_nonzero_lora_B_values=nonzero,production_accepted=False,answer_quality_accepted=False,
        passed=tensor_equal and runtime_equal and all(c['passed'] and c['argmax_equal'] and c['generation_equal'] for c in merge_checks)
            and all(c['logits_equal'] and c['generation_equal'] for c in reload_checks))
    (args.output/'export-checks.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise RuntimeError('FP32 merge or exact BF16 serialization check failed; not a release')
    weights={f.name:dict(sha256=sha256(f),bytes=inspect_safetensors(f)['bytes']) for f in args.output.glob('*.safetensors')}
    metadata={f.name:sha256(f) for f in args.output.glob('*.json')}
    manifest=dict(release='native-internal-v1',architecture='OlmoeForCausalLM',weights=weights,metadata_sha256=metadata,
        base=json.loads((base/'pinned_source.json').read_text()),training_run=str(args.run.resolve()),
        provenance_sha256=sha256(args.run/'provenance.json'),production_accepted=False,
        access='restricted_noncommercial_research',answer_quality_accepted=False,independent_answer_review='unassessed',
        required_setting='MOE_ALLOW_SAFETY_CANDIDATE=1',experts_and_gates_frozen=True,export_numeric_checks=report)
    (args.output/'release.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(export_passed=True,path=str(args.output))),flush=True)

if __name__=='__main__':main()
