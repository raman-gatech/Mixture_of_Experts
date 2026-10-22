"""Build substantive router/SFT corpora from pinned annotated public datasets.

BENIGN is a partial label: it means any non-safety domain, not GENERAL.
Official holdouts and normalized prompt groups are protected across both tasks.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from moe_router.data_io import read_jsonl

LABELS = ("MATH", "SCIENCE", "GENERAL", "SAFETY")


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def prompt_key(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def select_safe_response(row):
    """A 'safer' preference alone is insufficient: explicit safe flag is required."""
    preferred = row.get("better_response_id")
    safer = row.get("safer_response_id")
    for index in dict.fromkeys([preferred, safer, 0, 1]):
        if index not in (0, 1):
            continue
        response = row.get(f"response_{index}")
        if row.get(f"is_response_{index}_safe") is True and isinstance(response, str) and response.strip():
            return response.strip(), index
    return None


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prepare(raw, domains, destination, seed=42):
    if destination.exists():
        raise ValueError("Choose a new destination; prepared datasets are immutable")
    records, external, rejected = [], [], Counter()
    for split in ("train", "validation", "test"):
        for row in read_jsonl(domains / f"{split}.jsonl"):
            if row["label"] not in LABELS[:3]:
                raise ValueError("Unexpected legacy domain label")
            records.append(dict(task="router", split=split, text=row["text"], label=row["label"], source="original_domains"))
        for row in json.loads((raw / "aegis" / f"{split}.json").read_text()):
            text = row.get("prompt")
            if not isinstance(text, str) or not text.strip() or text.strip() == "REDACTED":
                rejected["aegis_redacted_or_empty"] += 1
                continue
            if row.get("prompt_label_source") != "human" or row.get("prompt_label") not in ("safe", "unsafe"):
                rejected["aegis_not_human_labelled"] += 1
                continue
            records.append(dict(task="router", split=split, text=text.strip(),
                                label="SAFETY" if row["prompt_label"] == "unsafe" else "BENIGN",
                                source="aegis", source_id=row["id"], categories=row.get("violated_categories", "")))
    for split in ("train", "test"):
        for row in read_jsonl(raw / "pku" / f"{split}.jsonl"):
            selected = select_safe_response(row)
            if selected is None or not row.get("prompt", "").strip():
                rejected["pku_no_explicit_safe_response"] += 1
                continue
            response, index = selected
            text = row["prompt"].strip()
            assigned = split
            if split == "train" and int(prompt_key(text)[:8], 16) % 10 == 0:
                assigned = "validation"
            records.append(dict(task="sft", split=assigned, instruction=text, input="", output=response,
                                source="pku", selected_response=index, explicitly_safe=True,
                                response_source=row.get(f"response_{index}_source")))
    with (raw / "do_not_answer" / "data_en.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            external.append(dict(text=row["question"], label="SAFETY", source="do_not_answer",
                                 categories=row["types_of_harm"], source_id=row["id"]))
    blocked = {prompt_key(row["text"]) for row in external}
    grouped = defaultdict(list)
    for record in records:
        text = record.get("text", record.get("instruction"))
        grouped[prompt_key(text)].append(record)
    prepared = {task: {name: [] for name in ("train", "validation", "test")} for task in ("router", "sft")}
    priority = {"train": 0, "validation": 1, "test": 2}
    for key, rows in grouped.items():
        if key in blocked:
            rejected["external_benchmark_overlap"] += len(rows)
            continue
        assigned = max((row["split"] for row in rows), key=priority.get)
        labels = {row["label"] for row in rows if row["task"] == "router" and row["label"] != "BENIGN"}
        incompatible = len(labels) > 1 or "SAFETY" in labels and any(row.get("label") == "BENIGN" for row in rows)
        if incompatible:
            rejected["conflicting_prompt_groups"] += 1
            continue
        for task in prepared:
            eligible = [row for row in rows if row["task"] == task and row["split"] == assigned]
            rejected["lower_priority_split_rows"] += sum(row["task"] == task and row["split"] != assigned for row in rows)
            if not eligible:
                continue
            if task == "router":
                eligible.sort(key=lambda row: row["label"] == "BENIGN")
            selected = dict(eligible[0], prompt_sha256=key)
            selected.pop("task")
            selected.pop("split")
            prepared[task][assigned].append(selected)
            rejected["duplicate_rows"] += len(eligible) - 1
    # Assert joint train/validation/test independence across router and generator.
    split_keys = {split: {row["prompt_sha256"] for task in prepared for row in prepared[task][split]}
                  for split in priority}
    if any(split_keys[a] & split_keys[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Cross-task split overlap")
    rng = random.Random(seed)
    counts = {}
    for task, splits in prepared.items():
        counts[task] = {}
        for name, rows in splits.items():
            if not rows:
                raise ValueError(f"Empty {task}/{name} split")
            rng.shuffle(rows)
            write_rows(destination / task / f"{name}.jsonl", rows)
            counts[task][name] = dict(Counter(row.get("label", "SAFE_RESPONSE") for row in rows))
    write_rows(destination / "evaluation/do_not_answer.jsonl", external)
    manifest = dict(seed=seed, labels=LABELS, partial_label="BENIGN: train non-safety probability without inventing a domain",
                    counts=counts, rejected=dict(rejected), external_evaluation_rows=len(external),
                    joint_split_overlap=0, source_manifest=json.loads((raw / "manifest.json").read_text()),
                    source_configs=json.loads(Path("configs/safety_sources.json").read_text()),
                    limitations=["Published SFT responses are model-generated with human safety/preference annotations, not human-written targets",
                                 "Earlier domain checkpoint saw original corpus; domain holdouts measure regression, not independent generalization",
                                 "Aegis categories concern the interaction; category-specific prompt metrics are proxies",
                                 "Human safety labels are not proof of alignment with every application policy detail"])
    # Hash the actual retained rows, including all protected holdouts.
    manifest["prepared_sha256"] = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in destination.rglob("*.jsonl")}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/safety/raw"))
    parser.add_argument("--domains", type=Path, default=Path("data/router/clean"))
    parser.add_argument("--output", type=Path, default=Path("data/safety/prepared/v1"))
    args = parser.parse_args()
    manifest = prepare(args.raw, args.domains, args.output)
    print(json.dumps({key: manifest[key] for key in ("counts", "rejected", "joint_split_overlap", "external_evaluation_rows")}, indent=2))
