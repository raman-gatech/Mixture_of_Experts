#!/usr/bin/env python3
"""Produce a blind-review table; model generation alone is not a quality score."""
import argparse
import csv
import json
from pathlib import Path
import random
from moe_router.backend import Backend
from moe_router.settings import Settings

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True, help="JSONL: text, label, optional reference answer")
parser.add_argument("--output", type=Path, default=Path("outputs/expert-comparison"))
parser.add_argument("--limit", type=int, default=30)
args = parser.parse_args()
backend = Backend(Settings.from_env()); backend.load()
rows = [json.loads(line) for line in args.data.read_text().splitlines()][:args.limit]
args.output.mkdir(parents=True, exist_ok=True)
results, review = [], []
rng = random.Random(42)
for number, row in enumerate(rows):
    candidates = []
    for mode in ["ROUTED", "BASE", "COMBINED", "MATH", "SCIENCE", "GENERAL"]:
        answer = backend.generate(row["text"], expert_override=None if mode == "ROUTED" else mode)
        result = {"question_id": number, "mode": mode, "expected_expert": row["label"], "reference": row.get("reference", ""), **answer}
        results.append(result); candidates.append(result)
    rng.shuffle(candidates)
    for index, answer in enumerate(candidates):
        answer["blind_candidate"] = index
        review.append({"question_id": number, "candidate": index, "prompt": row["text"], "reference": row.get("reference", ""), "answer": answer["response"], "correctness_0_to_2": "", "helpfulness_0_to_2": "", "notes": ""})
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
with (args.output / "blind-review.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(review[0]) if review else ["question_id"])
    writer.writeheader(); writer.writerows(review)
print("Review answers independently before claiming specialist gains:", args.output)
