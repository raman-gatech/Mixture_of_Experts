"""Bind raw cache diagnostics to an explicit, precision-aware adaptation gate."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--base-report',type=Path,required=True)
p.add_argument('--cache-report',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
base=json.loads(a.base_report.read_text())
cache=json.loads(a.cache_report.read_text())
assert cache['completed'] and len(cache['conditions'])==6
fp32=[step for c in cache['conditions'] if c['dtype']=='torch.float32' for step in c['steps']]
bf16=[step for c in cache['conditions'] if c['dtype']=='torch.bfloat16' for step in c['steps']]
assert len(fp32)==len(bf16)==12
policy=dict(version=2,fp32_max_abs=0.0001,bf16_max_kl_nats=0.01,
    require_all_greedy_tokens_equal=True,
    rationale='FP32 tests cache topology at tight tolerance. BF16 prefill/decode matrix shapes introduce rounding and sparse routing sensitivity; assess output distributions and greedy choices rather than a universal elementwise bound.',
    scope='Adaptation prerequisite on 3 prompts x 4 steps per precision; not exhaustive context or release acceptance',
    original_strict_bf16_check=base['cache'])
passed=(all(s['max_abs']<=policy['fp32_max_abs'] and s['argmax_equal'] for s in fp32)
    and all(s['kl_full_to_cached']<=policy['bf16_max_kl_nats'] and s['argmax_equal'] for s in bf16)
    and len(base['answers'])>=8 and all(x['answer'].strip() for x in base['answers']))
base.update(passed=passed,cache_precision_policy=policy,
    cache_diagnostics=dict(path=str(a.cache_report),sha256=hashlib.sha256(a.cache_report.read_bytes()).hexdigest()),
    original_qualification=dict(path=str(a.base_report),sha256=hashlib.sha256(a.base_report.read_bytes()).hexdigest()),
    answer_quality_accepted=False,production_accepted=False,
    intended_use='Proceed to bounded real-data adaptation profile; engine and product release gates remain separate')
a.output.write_text(json.dumps(base,indent=2)+'\n')
print(json.dumps(dict(adaptation_qualification_passed=passed,fp32_max_abs=max(s['max_abs'] for s in fp32),
                      bf16_max_kl=max(s['kl_full_to_cached'] for s in bf16))))
if not passed: raise SystemExit(1)
