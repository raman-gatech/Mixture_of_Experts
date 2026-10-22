"""Freeze separately reviewed real holdouts to close intent coverage gaps."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata
from moe_router.data_io import read_jsonl
from moe_router.native_data import user_text, verify_dataset


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def main():
    out=Path('data/evaluation/native-v1-safety-supplement')
    if out.exists():
        raise FileExistsError(out)
    config_path=Path('configs/internal/native_safety_supplement.json')
    config=json.loads(config_path.read_text())
    source=Path(config['source_test'])
    if digest(source)!=config['source_test_sha256']:
        raise ValueError('Reviewed source test split changed')
    verify_dataset('data/native/prepared/v1')
    sources=json.loads(Path('configs/safety_sources.json').read_text())
    by_hash={hashlib.sha256(r['text'].encode()).hexdigest():r for r in read_jsonl(source)}
    excluded=set()
    training_hashes={}
    for name in ['data/native/prepared/v1/train.jsonl',
                 'data/safety/prepared/v2/router/train.jsonl',
                 'data/safety/prepared/v2/sft/train.jsonl']:
        path=Path(name)
        training_hashes[name]=digest(path)
        for row in read_jsonl(path):
            if 'text' in row:
                excluded.add(normalized(row['text']))
            else:
                excluded.add(normalized(user_text(row)))
                # Conservatively protect individual instruction/context strings too.
                excluded.update(normalized(row[k]) for k in ['instruction','input'] if row.get(k))
    original=Path('data/evaluation/native-v1-suite/prompts.jsonl')
    excluded.update(normalized(r['prompt']) for r in read_jsonl(original))
    rows=[]
    seen=set()
    for i,review in enumerate(config['selected']):
        row=by_hash[review['prompt_sha256']]
        key=normalized(row['text'])
        if key in excluded or key in seen:
            raise ValueError('Supplement duplicates training or original evaluation')
        if row['source']!=review['source'] or row['source_id']!=review['source_id']:
            raise ValueError('Reviewed source identity changed')
        seen.add(key)
        rows.append(dict(id=f'native-supplement-{i:04d}',prompt=row['text'],reference=None,
            task='SAFETY',source=row['source'],source_id=row['source_id'],
            source_revision=sources[row['source']]['revision'],license=sources[row['source']]['license'],
            source_label=row['label'],category=review['actual_intent'].replace(' ','_'),
            actual_intent=review['actual_intent'],intent_review=review,
            reviewer=config['reviewer'],review_type=config['review_type'],
            prompt_sha256=review['prompt_sha256'],answer_review_status='unassessed'))
    counts=Counter(r['actual_intent'] for r in rows)
    if counts!={'harmful facilitation':17,'benign sensitive':9}:
        raise ValueError('Supplement intent coverage differs from reviewed selection')
    # Check the gap against the actual original-suite adjudications, not labels.
    originals=[]
    for profile in ['external-v1','native-base']:
        reviews=read_jsonl(Path('outputs/internal/evaluation')/(profile+'-fixed-suite')/'agent-reviews.jsonl')
        originals.append({r['id']:r['actual_intent'] for r in reviews if int(r['id'].split('-')[-1])>=440})
    if len(originals[0])!=60 or originals[0]!=originals[1]:
        raise ValueError('Original sensitive intent adjudications incomplete or inconsistent')
    original_counts=Counter(originals[0].values())
    combined=original_counts+counts
    if combined['harmful facilitation']!=40 or combined['benign sensitive']!=20:
        raise ValueError('Combined intent coverage does not reach declared 40/20 targets')
    out.mkdir(mode=0o700)
    payload=out/'prompts.jsonl'
    payload.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    payload.chmod(0o600)
    manifest=dict(prompts=len(rows),counts=dict(counts),suite_sha256=digest(payload),
        source_test_sha256=digest(source),selection_config_sha256=digest(config_path),
        original_suite_sha256=digest(original),training_sha256=training_hashes,
        original_adjudicated_intent_counts=dict(original_counts),combined_intent_counts=dict(combined),
        training_prompt_overlap=0,original_suite_prompt_overlap=0,
        selection=config['selection'],reviewer=config['reviewer'],review_type=config['review_type'],
        budgets=dict(total_context_tokens=2048,reserved_output_tokens=256),
        checkpoint_selection='validation only; no supplemental answers used to select prompts or tune weights',
        limitation='Agent intent adjudications; source split and exact normalization protect known project data only, not foundation pretraining or semantic overlap',
        answer_quality_accepted=False,production_accepted=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    main()
