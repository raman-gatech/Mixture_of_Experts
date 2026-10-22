"""Apply completed, attributable reviews to training rows only, failing closed.

Reviewer identity is declared, not authenticated by this tool. No automatic
model labels, blank forms, or holdout relabelling are accepted.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

from moe_router.data_io import read_jsonl

INTENTS = {"HARMFUL_FACILITATION", "SUPPORTIVE_DISTRESS", "RISKY_PERSONAL_ADVICE",
           "BENIGN_EDUCATION", "ORDINARY_DOMAIN", "UNCERTAIN"}
SPECIALIST = {"HARMFUL_FACILITATION", "SUPPORTIVE_DISTRESS", "RISKY_PERSONAL_ADVICE"}


def accepted_reviews(records, training_rows):
    by_key = {row["prompt_sha256"]: row for row in training_rows}
    accepted = {}
    for review in records:
        fields = ("reviewer_1_intent", "reviewer_2_intent", "adjudicated_intent")
        if not any(review.get(field, "").strip() for field in fields):
            continue
        digest = review["prompt_sha256"]
        if digest not in by_key or review.get("text") != by_key[digest]["text"]:
            raise ValueError("Reviewed prompts must exactly match training rows; holdouts cannot be imported")
        if digest in accepted:
            raise ValueError("Duplicate reviewed prompt")
        reviewers = [review.get(f"reviewer_{index}_id", "").strip() for index in (1, 2)]
        intents = [review.get(f"reviewer_{index}_intent", "").strip() for index in (1, 2)]
        reasons = [review.get(f"reviewer_{index}_reason", "").strip() for index in (1, 2)]
        if not all(reviewers) or len(set(reviewers)) != 2 or not all(reasons) or any(intent not in INTENTS for intent in intents):
            raise ValueError("Require two distinct attributed reviewers, reasons, and valid intents")
        adjudicated = review.get("adjudicated_intent", "").strip()
        if intents[0] != intents[1]:
            adjudicator = review.get("adjudicator_id", "").strip()
            if adjudicated not in INTENTS or not adjudicator or adjudicator in reviewers or not review.get("adjudicator_reason", "").strip():
                raise ValueError("Disagreement requires a distinct attributed adjudicator and reason")
            intent = adjudicated
        else:
            if adjudicated and adjudicated != intents[0]:
                raise ValueError("Adjudication cannot silently override agreeing reviewers")
            intent = intents[0]
        accepted[digest] = dict(intent=intent, reviewer_ids=reviewers,
                                adjudicator_id=review.get("adjudicator_id", ""))
    if not accepted:
        raise ValueError("No completed reviews; blank forms cannot become new training labels")
    return accepted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer-type", choices=("human", "agent"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new immutable output")
    original = read_jsonl(args.data / "router/train.jsonl")
    with args.reviews.open(newline="") as stream:
        reviews = accepted_reviews(list(csv.DictReader(stream)), original)
    revised = []
    counts = Counter()
    for row in original:
        review = reviews.get(row["prompt_sha256"])
        if review is None:
            revised.append(row)
            continue
        counts[review["intent"]] += 1
        if review["intent"] == "UNCERTAIN":
            continue
        label = "SAFETY" if review["intent"] in SPECIALIST else (row["label"] if row["label"] in ("MATH", "SCIENCE", "GENERAL") else "BENIGN")
        revised.append(dict(row, label=label, prior_label=row["label"], policy_review=review,
                            reviewer_type_declared=args.reviewer_type))
    shutil.copytree(args.data, args.output)
    with (args.output / "router/train.jsonl").open("w") as stream:
        for row in revised:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = json.loads((args.output / "manifest.json").read_text())
    manifest.update(policy_review=dict(reviewed_rows=len(reviews), intents=dict(counts),
                                        reviewers_declared=args.reviewer_type, identities_authenticated=False,
                                        csv_sha256=hashlib.sha256(args.reviews.read_bytes()).hexdigest(),
                                        training_only=True), old_training_rows_unchanged=False)
    manifest["counts"]["router"]["train"] = dict(Counter(row["label"] for row in revised))
    manifest["source_counts"]["router"]["train"] = dict(Counter(row["source"] for row in revised))
    manifest["prepared_sha256"]["router/train.jsonl"] = hashlib.sha256((args.output / "router/train.jsonl").read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["policy_review"]))
