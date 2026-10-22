"""Read-only CPU audit of exactly the supervised tokens in a pinned GPT-OSS run."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from moe_router.gptoss_format import encode_example, FINAL_PREFIX
from moe_router.native_data import verify_dataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    values = sorted(values)
    return {str(p): values[round((len(values)-1)*p)] for p in (0, .5, .9, .99, 1)} if values else {}


def audit(run, tokenizer):
    run = Path(run)
    provenance = json.loads((run/'provenance.json').read_text())
    identity = provenance['identity']; cfg = identity['config']; data = Path(cfg['data'])
    verify_dataset(data)
    if digest(data/'manifest.json') != identity['dataset_manifest_sha256']:
        raise ValueError('Dataset provenance mismatch')
    for name, expected in identity['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError(f'Active encoding source changed: {name}')
    header = tokenizer.encode(FINAL_PREFIX, add_special_tokens=False)
    groups = defaultdict(lambda: dict(rows=0, prompt_tokens=[], response_tokens=[],
                                     total_tokens=[], over_output_budget=0, header_mismatch=0,
                                     missing_end_of_turn=0, retained_sources=Counter()))
    retained = Counter(); rejected = Counter(); flags = []; targets = defaultdict(list)
    for line in (data/'train.jsonl').open():
        row = json.loads(line)
        encoded, reason = encode_example(row, tokenizer, cfg['max_length'], 'mixed')
        if reason:
            rejected[row['source']+':'+reason] += 1
            continue
        source = row['source']; retained[source] += 1
        labels = encoded['labels']; first = next(i for i, label in enumerate(labels) if label != -100)
        # A noncontiguous mask or shifted response would train on user/system input.
        if labels[:first] != [-100]*first or labels[first:] != encoded['input_ids'][first:]:
            raise ValueError('Supervision is not a contiguous assistant-only suffix')
        group = groups[source]; response = labels[first:]
        group['rows'] += 1; group['prompt_tokens'].append(first)
        group['response_tokens'].append(len(response)); group['total_tokens'].append(len(labels))
        header_bad = response[:len(header)] != header
        end_bad = response[-1] != tokenizer.eos_token_id
        group['header_mismatch'] += int(header_bad); group['missing_end_of_turn'] += int(end_bad)
        # Serving pre-fills the final header; it does not consume generated-token budget.
        generated_target_tokens = len(response)-len(header)
        too_long = generated_target_tokens > cfg['output_limit']
        group['over_output_budget'] += int(too_long)
        if too_long or header_bad or end_bad:
            flags.append(dict(source=source, source_id=row['source_id'],
                              prompt_sha256=row['prompt_sha256'], target_sha256=hashlib.sha256(row['output'].encode()).hexdigest(),
                              generated_target_tokens=generated_target_tokens,
                              over_output_budget=too_long, header_mismatch=header_bad, missing_end_of_turn=end_bad))
        targets[(source, hashlib.sha256(row['output'].encode()).hexdigest())].append(row['source_id'])
    if dict(retained) != provenance['train']['retained_by_source'] or dict(rejected) != provenance['train']['rejected']:
        raise ValueError('Actual tokenization no longer reproduces training eligibility')
    total_response = sum(sum(group['response_tokens']) for group in groups.values())
    result = {}
    for source, group in groups.items():
        result[source] = dict(rows=group['rows'],
            prompt_tokens_quantiles=quantiles(group['prompt_tokens']),
            response_tokens_quantiles=quantiles(group['response_tokens']),
            total_tokens_quantiles=quantiles(group['total_tokens']),
            supervised_tokens=sum(group['response_tokens']),
            supervised_token_share=sum(group['response_tokens'])/total_response,
            targets_exceeding_serving_output_budget=group['over_output_budget'],
            header_mismatch=group['header_mismatch'], missing_end_of_turn=group['missing_end_of_turn'])
    return dict(run=str(run), provenance_sha256=digest(run/'provenance.json'),
        dataset_manifest_sha256=identity['dataset_manifest_sha256'],
        encoding_source_sha256=identity['source_sha256'], output_limit=cfg['output_limit'], max_length=cfg['max_length'],
        retained_by_source=dict(retained), rejected=dict(rejected), sources=result, flagged_targets=flags,
        duplicate_target_groups=[dict(source=s, target_sha256=h, source_ids=ids) for (s,h),ids in targets.items() if len(ids)>1],
        scope='CPU tokenization audit only: lengths and duplicate targets do not establish causality or factual quality.',
        training_targets_modified=False, active_training_modified=False, production_accepted=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); os.umask(0o077)
    if args.output.exists(): raise FileExistsError(args.output)
    from transformers import AutoTokenizer
    cfg=json.loads((args.run/'provenance.json').read_text())['identity']['config']
    tokenizer=AutoTokenizer.from_pretrained(cfg['base'],local_files_only=True)
    result=audit(args.run,tokenizer)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('flagged_targets','duplicate_target_groups','encoding_source_sha256')}))


if __name__=='__main__': main()
