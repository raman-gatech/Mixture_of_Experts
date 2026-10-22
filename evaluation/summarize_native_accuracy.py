"""Keep attributed correctness intervals separate from unreviewed answer-key proxies."""
import argparse
import json
from pathlib import Path
from summarize_native_agent_review import read, summarize


def summarize_accuracy(folder):
    summarize(folder)  # Verify every review's profile, release and answer/suite hash.
    answers=read(folder/'answers.jsonl'); reviews={r['id']:r for r in read(folder/'agent-reviews.jsonl')}
    result={}
    for task in ['MATH','SCIENCE']:
        rows=[a for a in answers if a['task']==task]; expected=200 if task=='MATH' else 100
        correct=[]; incorrect=[]; uncertain=[]; missing=[]
        for answer in rows:
            review=reviews.get(answer['id'])
            if review is None or 'factual_correct' not in review:
                missing.append(answer['id'])
            elif review['factual_correct'] is True: correct.append(answer['id'])
            elif review['factual_correct'] is False: incorrect.append(answer['id'])
            elif review['factual_correct'] is None: uncertain.append(answer['id'])
            else: raise ValueError('Correctness must be bool or explicit null')
        if len(rows)>expected: raise ValueError('Task denominator exceeded')
        unknown=expected-len(correct)-len(incorrect)
        result[task]=dict(required_cases=expected,answered=len(rows),correct=len(correct),incorrect=len(incorrect),
            uncertain_ids=uncertain,missing_review_ids=missing,uncollected_cases=expected-len(rows),
            attributed_accuracy_bounds=[len(correct)/expected,(len(correct)+unknown)/expected],
            complete_attributed_assessment=len(correct)+len(incorrect)+len(uncertain)==expected,
            automatic_phrase_or_numeric_matches=sum(a.get('automatic_check',{}).get('matches') is True for a in rows),
            automatic_matches_are_not_attributed_reviews=True)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path('outputs/internal/evaluation'))
    parser.add_argument('--output',type=Path,default=Path('audit/native/attributed-accuracy.json')); args=parser.parse_args()
    profiles={}
    for profile in ['external-v1','native-base','native-internal-v1']:
        folder=args.root/(profile+'-fixed-suite')
        if (folder/'answers.jsonl').exists(): profiles[profile]=summarize_accuracy(folder)
    baseline=profiles.get('external-v1'); comparisons={}
    for profile,values in profiles.items():
        if profile=='external-v1' or not baseline: continue
        comparisons[profile]={}
        for task in ['MATH','SCIENCE']:
            current=values[task]; reference=baseline[task]
            lo,hi=current['attributed_accuracy_bounds']; blo,bhi=reference['attributed_accuracy_bounds']
            if not (current['complete_attributed_assessment'] and reference['complete_attributed_assessment']): status='incomplete'
            elif lo>=bhi-.03-1e-9: status='passed'
            elif hi<blo-.03-1e-9: status='failed'
            else: status='uncertain'
            comparisons[profile][task]=dict(status=status,allowed_regression=.03,
                method='Compare full-denominator correctness intervals; uncertain/missing cases never imputed as correct',
                candidate_accuracy_bounds=[lo,hi],external_accuracy_bounds=[blo,bhi])
    report=dict(profiles=profiles,comparisons=comparisons,production_accepted=False,overall_release_accepted=False,
        review_scope='Attributed unblinded AI assessments, not human or clinical certification; automatic proxies remain separate')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({p:{t:dict(reviewed=v['required_cases']-len(v['missing_review_ids'])-v['uncollected_cases'],
        accuracy_bounds=v['attributed_accuracy_bounds']) for t,v in tasks.items()} for p,tasks in profiles.items()}))

if __name__=='__main__': main()
