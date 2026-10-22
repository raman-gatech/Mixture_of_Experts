#!/usr/bin/env python3
"""Deduplicate before splitting; quarantine ambiguous labels rather than guessing."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import random
import re


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def prepare(paths, destination, seed=42):
    groups = defaultdict(list)
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not isinstance(row.get("text"), str) or not row["text"].strip() or row.get("label") not in {"MATH", "SCIENCE", "GENERAL"}:
                raise ValueError(f"Invalid row in {path}")
            groups[normalize(row["text"])].append(row)
    conflicts = [rows for rows in groups.values() if len({row["label"] for row in rows}) > 1]
    unique = [rows[0] for rows in groups.values() if len({row["label"] for row in rows}) == 1]
    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for label in ("MATH", "SCIENCE", "GENERAL"):
        rows = sorted((r for r in unique if r["label"] == label), key=lambda r: normalize(r["text"]))
        rng.shuffle(rows)
        train_end, validation_end = int(len(rows) * .8), int(len(rows) * .9)
        for name, subset in zip(splits, (rows[:train_end], rows[train_end:validation_end], rows[validation_end:])):
            splits[name].extend(subset)
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows)
        (destination / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (destination / "conflicts.json").write_text(json.dumps(conflicts, indent=2) + "\n")
    report = {"seed": seed, "input_rows": sum(map(len, groups.values())), "unique_retained": len(unique), "conflicting_question_groups": len(conflicts), "splits": {name: dict(Counter(r["label"] for r in rows)) for name, rows in splits.items()}, "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, "limitation": "These new splits are for NEW training. Recovered checkpoints may have seen these questions; evaluate those on independently sourced data."}
    (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", default=[Path("data/router/original/train.jsonl"), Path("data/router/original/test.jsonl")])
    parser.add_argument("--output", type=Path, default=Path("data/router/clean"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.seed), indent=2))
