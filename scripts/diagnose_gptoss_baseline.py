"""Compare runtime parity on previously changed validation answers; never alter gates."""
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from peft import LoraConfig, get_peft_model
    from moe_router.gptoss_format import prompt_ids, final_text
    parent = Path('outputs/internal/native/gptoss/20261018-r2-distributed/sft')
    failed = Path('outputs/internal/native/gptoss/20261018-r3-calibrated/sft')
    provenance = json.loads((parent / 'provenance.json').read_text())
    cfg = provenance['identity']['config']
    old = json.loads((parent / 'answer-validation-0.json').read_text())['answers']
    new = json.loads((failed / 'answer-validation-base-reproduced.json').read_text())['answers']
    keys = [(a['source'], a['source_id']) for a, b in zip(old, new)
            if a['automatic_match'] != b['automatic_match']]
    lookup = {(r['source'], r['source_id']): r for r in
              (json.loads(line) for line in (Path(cfg['data']) / 'validation.jsonl').open())}
    reference = {(r['source'], r['source_id']): r for r in old}
    torch.manual_seed(cfg['seed']); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(cfg['base'], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(cfg['base'], local_files_only=True,
        torch_dtype=torch.bfloat16, quantization_config=Mxfp4Config(dequantize=True),
        device_map={'': 0}, attn_implementation='eager').eval()
    output = Path('audit/native') / ('gptoss-baseline-parity-' + os.environ['SLURM_JOB_ID'])
    output.mkdir(mode=0o700)
    for mode in ['bare-tf32-off', 'bare-tf32-on', 'zero-lora-tf32-off']:
        torch.backends.cuda.matmul.allow_tf32 = mode == 'bare-tf32-on'
        if mode == 'zero-lora-tf32-off':
            torch.manual_seed(cfg['seed'])
            model = get_peft_model(model, LoraConfig(r=cfg['rank'], lora_alpha=cfg['alpha'],
                lora_dropout=.05, target_modules=cfg['target_modules'], task_type='CAUSAL_LM')).eval()
            assert all(torch.count_nonzero(p) == 0 for n, p in model.named_parameters() if 'lora_B' in n)
        answers = []
        began = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for key in keys:
                ids = torch.tensor([prompt_ids(tokenizer, lookup[key])], device='cuda')
                generated = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=256,
                    do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                answer = final_text(tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=False))
                answers.append(dict(source=key[0], source_id=key[1], answer=answer,
                    identical_to_original=answer == reference[key]['answer']))
                print(json.dumps(dict(mode=mode, cases=len(answers), source_id=key[1],
                    identical_to_original=answers[-1]['identical_to_original'])), flush=True)
        (output / (mode + '.json')).write_text(json.dumps(dict(mode=mode, answers=answers,
            elapsed_seconds=time.monotonic()-began, job=os.environ['SLURM_JOB_ID'],
            production_accepted=False), indent=2)+'\n')


if __name__ == '__main__':
    os.umask(0o077)
    main()
