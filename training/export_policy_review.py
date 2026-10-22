"""Export real training prompts for policy annotation; never invent training labels."""
import argparse
import csv
import json
from pathlib import Path

from moe_router.data_io import read_jsonl

CATEGORIES = {'Suicide and Self Harm', 'Unauthorized Advice', 'Needs Caution'}
LABELS = ('HARMFUL_FACILITATION', 'SUPPORTIVE_DISTRESS', 'RISKY_PERSONAL_ADVICE',
          'BENIGN_EDUCATION', 'ORDINARY_DOMAIN', 'UNCERTAIN')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/safety/prepared/v1'))
    parser.add_argument('--output', type=Path, default=Path('data/safety/prepared/v1/policy-review'))
    parser.add_argument('--show-source-labels', action='store_true', help='Default review forms hide source labels/categories to reduce anchoring')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    rows = read_jsonl(args.data / 'router/train.jsonl')
    selected = [row for row in rows if row['source'] == 'aegis' and
                (CATEGORIES & {value.strip() for value in row.get('categories', '').split(',')})]
    args.output.mkdir(parents=True)
    args.output.chmod(0o700)
    columns = ('prompt_sha256', 'source_id', 'text',
               'reviewer_1_intent', 'reviewer_1_reason', 'reviewer_1_id',
               'reviewer_2_intent', 'reviewer_2_reason', 'reviewer_2_id', 'adjudicated_intent',
               'adjudicator_id', 'adjudicator_reason')
    if args.show_source_labels:
        columns += ('original_label', 'interaction_categories')
    with (args.output / 'training_prompts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in sorted(selected, key=lambda item: item['prompt_sha256']):
            item = dict(prompt_sha256=row['prompt_sha256'], source_id=row['source_id'], text=row['text'])
            if args.show_source_labels:
                item.update(original_label=row['label'], interaction_categories=row.get('categories', ''))
            writer.writerow(item)
    (args.output / 'instructions.json').write_text(json.dumps(dict(
        status='Unlabelled review queue, never an automatic training input', rows=len(selected),
        source_labels_hidden=not args.show_source_labels,
        source_split='train only; validation/test and external benchmarks unchanged',
        labels=LABELS, proposed_routing={'HARMFUL_FACILITATION':'SAFETY', 'SUPPORTIVE_DISTRESS':'SAFETY',
                                         'RISKY_PERSONAL_ADVICE':'SAFETY', 'BENIGN_EDUCATION':'original domain if known',
                                         'ORDINARY_DOMAIN':'original domain if known', 'UNCERTAIN':'quarantine/adjudicate'},
        instructions=['Judge the visible user prompt, not the attached interaction category or original unsafe label.',
                      'Harmful facilitation requests help carrying out abuse, violence, exploitation, self-harm, crime, or privacy invasion.',
                      'Supportive distress seeks help with personal crisis, abuse, or emotional suffering; a benign safety label does not imply ordinary domain routing.',
                      'Risky personal advice requests individualized medical or other high-stakes decisions; distinguish general education.',
                      'Do not mark a factual educational request harmful merely because it mentions a sensitive topic.',
                      'Record uncertainty and independent reviewer disagreement. Obtain qualified review for clinical/supportive cases.',
                      'Do not modify protected holdouts or accept generated labels as independent human ground truth.']), indent=2) + '\n')
    print(json.dumps({'review_prompts':len(selected), 'output':str(args.output), 'training_labels_created':0}))


if __name__ == '__main__':
    main()
