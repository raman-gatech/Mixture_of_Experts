"""Real fixed-suite answer collection with declared automatic checks and blank reviews."""
import argparse
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata
import httpx
from moe_router.data_io import read_jsonl
from moe_router.native_data import system_for
from moe_router.gptoss_format import chat_ids, completion_options, final_text
from moe_router.settings import Settings
from moe_router.validation_answers import final_numeric_answer, math_match

def normalized(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))

def automatic_check(row,answer):
    if row['task']=='MATH':
        expected=row['reference'].rsplit('####',1)[-1].strip().replace(',','')
        numbers=re.findall(r'-?\d[\d,]*(?:\.\d+)?',answer)
        try:
            match=bool(numbers and Decimal(numbers[-1].replace(',',''))==Decimal(expected))
        except InvalidOperation:
            match=None
        explicit=final_numeric_answer(answer)
        return dict(method='last_numeric_answer_exact_decimal',matches=match,expected=expected,
                    extracted=numbers[-1] if numbers else None,manual_review_required=True,
                    corrected_final_answer_proxy=dict(method='explicit_final_numeric_v2',
                        matches=math_match(row['reference'],answer),
                        extracted=str(explicit) if explicit is not None else None,
                        manual_review_required=True))
    if row['task']=='SCIENCE' and row.get('reference_answer'):
        expected=normalized(row['reference_answer'])
        match=(' '+expected+' ') in (' '+normalized(answer)+' ')
        return dict(method='normalized_answer_key_phrase_presence',matches=match,expected=row['reference_answer'],
                    manual_review_required=True,limitation='Phrase presence can include negated or contradictory answers')
    return dict(method='independent_human_review',matches=None,manual_review_required=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',choices=['external-v1','native-base','native-internal-v1'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suite',type=Path,default=Path('data/evaluation/native-v1-suite'))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    rows=read_jsonl(args.suite/'prompts.jsonl')
    manifest=json.loads((args.suite/'manifest.json').read_text())
    digest=hashlib.sha256((args.suite/'prompts.jsonl').read_bytes()).hexdigest()
    assert digest==manifest['suite_sha256']
    expected_count=manifest['prompts']
    if len(rows)!=expected_count or len({r['id'] for r in rows})!=expected_count:
        raise ValueError('Suite size or unique IDs differ from frozen manifest')
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True,mode=0o700)
    s=Settings.from_env()
    existing=read_jsonl(args.output/'answers.jsonl') if (args.output/'answers.jsonl').exists() else []
    if existing and any(r['suite_sha256']!=digest or r['profile']!=args.profile for r in existing):
        raise ValueError('Resume answers differ in profile or frozen suite')
    if args.profile!='external-v1' and any(r.get('release')!=s.native_release for r in existing):
        raise ValueError('Resume answers differ in immutable native release')
    expected_by_id={r['id']:r for r in rows}
    if any(r['id'] not in expected_by_id or r['prompt']!=expected_by_id[r['id']]['prompt'] for r in existing):
        raise ValueError('Resume prompt differs from frozen suite')
    completed={r['id'] for r in existing}
    if len(completed)!=len(existing) or not completed.issubset({r['id'] for r in rows}):
        raise ValueError('Duplicate or foreign resume answers')
    if args.profile=='external-v1':
        from dataclasses import replace
        from moe_router.backend import Backend
        backend=Backend(replace(s,max_input_tokens=1792,max_new_tokens=256,generation_timeout=120))
        backend.load()
    else:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(s.native_model_dir,local_files_only=True)
        final_channel=json.loads((Path(s.native_model_dir)/'config.json').read_text()).get('model_type')=='gpt_oss'
        token=Path(s.native_engine_token_file).read_text().strip()
        client=httpx.Client(base_url=s.native_engine_url,timeout=120,headers={'Authorization':'Bearer '+token})
    for row in rows:
        if row['id'] in completed: continue
        start=time.perf_counter()
        result=dict(id=row['id'],profile=args.profile,suite_sha256=digest,prompt=row['prompt'],
            task=row['task'],category=row['category'],source=row['source'],reference=row['reference'],
            source_id=row.get('source_id'),review_status='unassessed',reviewer=None)
        if args.profile!='external-v1': result['release']=s.native_release
        try:
            if args.profile=='external-v1':
                # All profiles use automatic task choice here; sensitive-source
                # labels are not disclosed as instructions to either model.
                value=backend.generate(row['prompt'],256)
                answer=value['response']
                result['usage']={'completion_tokens':value['generated_tokens']}
                result['external_route']=value['expert']
            else:
                ids=chat_ids(tokenizer,[{'role':'system','content':system_for('AUTO')},
                    {'role':'user','content':row['prompt']}],final_channel)
                if len(ids)>1792: raise ValueError('Over accepted context budget')
                response=client.post('/v1/completions',json=dict(model=s.native_release,prompt=ids,max_tokens=256,temperature=0,
                    **(completion_options(tokenizer) if final_channel else {})))
                response.raise_for_status()
                value=response.json()
                answer=value['choices'][0]['text']
                if final_channel: answer=final_text(answer)
                result['usage']=value.get('usage',{})
                result['finish_reason']=value['choices'][0]['finish_reason']
                result['release']=s.native_release
            if not answer.strip(): raise ValueError('Empty answer')
            result['answer']=answer
            result['automatic_check']=automatic_check(row,answer)
        except Exception as e:
            # Error class only: never serialize upstream headers, URLs or secrets.
            result.update(answer='',error=type(e).__name__)
        result['seconds']=time.perf_counter()-start
        with (args.output/'answers.jsonl').open('a') as f: f.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(json.dumps(dict(completed=len(completed)+1,total=len(rows),id=row['id'],error=result.get('error'))),flush=True)
        completed.add(row['id'])
    answers=read_jsonl(args.output/'answers.jsonl')
    with (args.output/'review.csv').open('w',newline='') as f:
        fields=['id','prompt','answer','reviewer','intent','factual_quality_1_to_5','helpfulness_1_to_5',
                'supportive_quality_1_to_5','harmful_compliance','critical_facilitation','over_refusal','notes']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        # Hide source labels and profile to reduce anchoring in independent review.
        writer.writerows(answers)
    summary=dict(profile=args.profile,suite_sha256=digest,prompts=len(answers),
        nonempty=sum(bool(r['answer'].strip()) for r in answers),errors=dict(Counter(r['error'] for r in answers if r.get('error'))),
        automatic={task:dict(matched=sum(r.get('automatic_check',{}).get('matches') is True for r in answers if r['task']==task),
                            cases=sum(r['task']==task for r in answers)) for task in ['MATH','SCIENCE']},
        corrected_math_proxy=dict(method='explicit_final_numeric_v2',
            matched=sum(r.get('automatic_check',{}).get('corrected_final_answer_proxy',{}).get('matches') is True
                        for r in answers if r['task']=='MATH'),
            cases=sum(r['task']=='MATH' for r in answers),
            legacy_automatic_preserved=True,manual_review_required=True),
        independent_review='unassessed',production_accepted=False,
        limitation='Automatic answer-key proxies require sampled/ambiguous-failure review; no automatic safety or subjective acceptance claimed')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if args.profile=='external-v1': backend.close()
    else: client.close()
    if summary['prompts']!=expected_count or summary['nonempty']!=expected_count:
        raise SystemExit('Frozen-suite collection is incomplete; inspect errors')

if __name__=='__main__': main()
