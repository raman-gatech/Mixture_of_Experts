"""Freeze 500 real heldout tasks; safety intent/review scores remain unassessed."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata
from moe_router.data_io import read_jsonl
from moe_router.native_data import verify_dataset, user_text

out=Path('data/evaluation/native-v1-suite')
if out.exists(): raise FileExistsError(out)
verify_dataset('data/native/prepared/v1')
rows=read_jsonl('data/native/prepared/v1/test.jsonl')
router=read_jsonl('data/safety/prepared/v2/router/test.jsonl')
def normalize(text): return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
train={normalize(user_text(r)) for r in read_jsonl('data/native/prepared/v1/train.jsonl')}
seen=set()
excluded=Counter()
def pick(rows,n):
    selected=[]
    for row in sorted(rows,key=lambda r:hashlib.sha256(('native-v1-suite-42'+r.get('prompt_sha256',r.get('text',''))).encode()).hexdigest()):
        key=normalize(row['text'] if 'text' in row else user_text(row))
        if key in train or key in seen:
            excluded['training_overlap' if key in train else 'duplicate_across_suite']+=1
            continue
        selected.append(row)
        seen.add(key)
        if len(selected)==n: break
    if len(selected)!=n: raise ValueError('Insufficient real holdout rows')
    return selected
selected=[]
for task,count in [('MATH',200),('SCIENCE',100),('GENERAL',100)]:
    for row in pick([r for r in rows if r['task']==task],count):
        selected.append(dict(prompt=user_text(row),reference=row['output'],reference_answer=row.get('reference_answer'),
            task=task,source=row['source'],source_id=row['source_id'],source_revision=row['source_revision'],
            review_status='unassessed',category=task.lower()))
for row in pick([r for r in rows if r['source']=='esconv'],40):
    selected.append(dict(prompt=user_text(row),reference=row['output'],task='SAFETY',source='esconv',
                         source_id=row['source_id'],source_revision=row['source_revision'],category='supportive',review_status='unassessed'))
for label,n,category in [('SAFETY',40,'publisher_harmful'),('BENIGN',20,'publisher_benign_proxy')]:
    for row in pick([r for r in router if r['label']==label],n):
        selected.append(dict(prompt=row['text'],reference=None,task='SAFETY',source=row.get('source'),
            source_id=row.get('source_id'),category=category,source_label=row['label'],
            source_categories=row.get('categories'),review_status='unassessed',
            limitation='Source label is not independent prompt-intent review; benign proxy may not be sensitive'))
prompts=[normalize(r['prompt']) for r in selected]
if len(set(prompts))!=500 or set(prompts)&train:
    raise ValueError('Release-suite duplicate or training overlap')
out.mkdir(mode=0o700)
for i,row in enumerate(selected):
    row['id']=f'native-v1-{i:04d}'
    row['prompt_sha256']=hashlib.sha256(row['prompt'].encode()).hexdigest()
with (out/'prompts.jsonl').open('x') as f:
    for row in selected: f.write(json.dumps(row,ensure_ascii=False)+'\n')
manifest=dict(prompts=500,counts=dict(Counter(r['category'] for r in selected)),seed=42,
    suite_sha256=hashlib.sha256((out/'prompts.jsonl').read_bytes()).hexdigest(),
    native_test_sha256=hashlib.sha256(Path('data/native/prepared/v1/test.jsonl').read_bytes()).hexdigest(),
    safety_test_sha256=hashlib.sha256(Path('data/safety/prepared/v2/router/test.jsonl').read_bytes()).hexdigest(),
    training_prompt_overlap=0,production_accepted=False,independent_review='unassessed',
    excluded_during_selection=dict(excluded),
    limitation='First frozen project comparison suite, not untouched foundation pretraining data. Benign intent/sensitivity requires independent review.',
    budgets=dict(total_context_tokens=2048,reserved_output_tokens=256),
    checkpoint_selection='validation only; release suite is not a tuning signal')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest),flush=True)
