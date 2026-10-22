"""Prepare provenance-bound, answer-bearing native SFT data without generating targets.

Only exact normalized overlap is checked; this is not semantic decontamination.
Published source annotations do not constitute a project safety/quality review.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from moe_router.data_io import read_jsonl

SPLITS = ("train", "validation", "test")


def key(text):
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
    return hashlib.sha256(text.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_keys(row):
    instruction = row.get("instruction") or row.get("text") or row.get("prompt") or row.get("question")
    if not isinstance(instruction, str) or not instruction.strip():
        return set()
    full = instruction + ("\n\n" + row["input"] if row.get("input") else "")
    return {key(instruction), key(full)}


def verify_inputs(raw, sources, safety):
    manifest = json.loads((raw / "manifest.json").read_text())
    found = set()
    for entry in manifest:
        source = sources[entry["source"]]
        filename = Path(entry["file"]).name
        allowed = {Path(name).name for name in source["files"]}
        if filename not in allowed or entry["revision"] != source["revision"] or entry["license"] != source["license"]:
            raise ValueError("Raw manifest differs from pinned source configuration")
        path = raw / entry["source"] / filename
        if sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Raw data integrity failure: {path}")
        found.add((entry["source"], filename))
    expected = {(name, Path(filename).name) for name, source in sources.items() for filename in source["files"]}
    if found != expected:
        raise ValueError("Raw manifest does not cover every pinned source file")
    safety_manifest = json.loads((safety / "manifest.json").read_text())
    for split in SPLITS:
        filename = f"sft/{split}.jsonl"
        if sha(safety / filename) != safety_manifest["prepared_sha256"][filename]:
            raise ValueError(f"Safety prepared data changed: {filename}")
    return manifest


def candidates(raw, safety, sources, safety_sources):
    result = []

    def add(instruction, output, task, source, split, source_id, context="", group=None, **metadata):
        result.append(dict(instruction=instruction, input=context, output=output, task=task,
                           source=source, split=split, source_id=source_id, group_id=group,
                           source_revision=(sources if source in sources else safety_sources)[source]["revision"],
                           license=(sources if source in sources else safety_sources)[source]["license"],
                           review_status="source_annotated_not_project_reviewed", **metadata))

    for official in ("train", "test"):
        for index, row in enumerate(read_jsonl(raw / "gsm8k" / f"{official}.jsonl")):
            split = "test" if official == "test" else ("validation" if int(key(row["question"])[:8], 16) % 10 == 0 else "train")
            add(row["question"], row["answer"], "MATH", "gsm8k", split, f"{official}:{index}",
                target_origin="published_human_answer_and_rationale")
    import pyarrow.parquet as parquet
    for split in SPLITS:
        path = raw / "sciq" / f"{split}-00000-of-00001.parquet"
        for index, row in enumerate(parquet.read_table(path).to_pylist()):
            support = row["support"].strip()
            answer = row["correct_answer"].strip()
            if support:
                answer += "\n\n" + support
            add(row["question"], answer, "SCIENCE", "sciq", split, f"{split}:{index}",
                group=f"sciq-support:{key(support)}" if support else None,
                target_origin="exact_correct_answer_plus_published_support", reference_answer=row["correct_answer"])
    for index, row in enumerate(read_jsonl(raw / "dolly" / "databricks-dolly-15k.jsonl")):
        context = row["context"].strip()
        add(row["instruction"], row["response"], "GENERAL", "dolly", None, str(index), context=context,
            group=f"dolly-context:{key(context)}" if context else None,
            target_origin="published_human_response", category=row["category"])
    for split in SPLITS:
        for index, row in enumerate(read_jsonl(safety / "sft" / f"{split}.jsonl")):
            support = (row.get("source") == "esconv" and row.get("human_written") is True and
                       row.get("target_eligibility") == "human_support_positive_feedback" and
                       row.get("positive_feedback", 0) >= 4 and row.get("conversation_sha256"))
            safe = row.get("source") == "pku" and row.get("explicitly_safe") is True
            if not (support or safe):
                raise ValueError("Safety targets require supported response annotations")
            evidence = {name: row[name] for name in ("explicitly_safe", "human_written", "positive_feedback",
                        "clinical_safety_reviewed", "target_eligibility", "response_source", "selected_response") if name in row}
            add(row["instruction"], row["output"], "SAFETY", row["source"], split, f"{split}:{index}",
                context=row.get("input", ""),
                group=f'esconv:{row["conversation_sha256"]}' if support else None,
                target_origin="participant_rated_human_support" if support else "human_safe_annotated_model_response",
                **evidence)
    return result


def partition(rows, blocked_train, cap=6000):
    """Connect prompt/context/conversation identities before any split decision."""
    if cap < 1:
        raise ValueError("Training task cap must be positive")
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        identities = {"prompt:" + digest for digest in prompt_keys(row)}
        if row.get("group_id"):
            identities.add("group:" + row["group_id"])
        for identity in identities:
            if identity in seen:
                parents[root(i)] = root(seen[identity])
            seen[identity] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    output = {split: [] for split in SPLITS}
    quarantine = []
    counts = Counter()
    for group in groups.values():
        reason = None
        if any(not isinstance(r.get("output"), str) or not r["output"].strip() or not prompt_keys(r) for r in group):
            reason = "empty_prompt_or_response"
        fixed = {r["split"] for r in group if r["split"]}
        if len(fixed) > 1:
            reason = "publisher_or_preserved_split_conflict"
        identities = set.union(*(prompt_keys(row) for row in group))
        split = next(iter(fixed)) if len(fixed) == 1 else None
        if split is None:
            bucket = int(min(identities)[:8], 16) % 10 if identities else 0
            split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        if split == "train" and identities & blocked_train:
            reason = "protected_existing_holdout"
        # Context-sharing questions may differ; identical full prompts must not
        # acquire contradictory targets. Quarantine instead of picking a winner.
        targets = defaultdict(set)
        for row in group:
            instruction = row.get("instruction")
            full = (instruction if isinstance(instruction, str) else "") + ("\n\n" + row["input"] if row.get("input") else "")
            answer = row.get("output")
            targets[key(full)].add(key(answer if isinstance(answer, str) else ""))
        if any(len(values) > 1 for values in targets.values()):
            reason = "conflicting_targets"
        if reason:
            counts[reason] += len(group)
            quarantine.extend(dict(source=r["source"], source_id=r["source_id"], reason=reason,
                                   prompt_sha256=sorted(prompt_keys(r))) for r in group)
            continue
        deduped = set()
        for row in sorted(group, key=lambda r: (r["source"], r["source_id"])):
            full = row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
            digest = key(full)
            if digest in deduped:
                counts["exact_duplicate"] += 1
                continue
            deduped.add(digest)
            output[split].append(dict(row, split=split, prompt_sha256=digest))
    # Keep every eligible supportive pair before sampling the larger safe-PKU
    # source. All sampling is training-only; holdouts are never balanced.
    train = sorted(output["train"], key=lambda r: (r["task"], r["source"] != "esconv", r["prompt_sha256"]))
    kept, tasks = [], Counter()
    for row in train:
        if tasks[row["task"]] >= cap:
            counts["training_task_cap"] += 1
            continue
        tasks[row["task"]] += 1
        kept.append(row)
    output["train"] = kept
    for split in SPLITS:
        output[split].sort(key=lambda r: r["prompt_sha256"])
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        a_keys = set().union(*(prompt_keys(r) for r in output[a]))
        b_keys = set().union(*(prompt_keys(r) for r in output[b]))
        a_groups = {r["group_id"] for r in output[a] if r.get("group_id")}
        b_groups = {r["group_id"] for r in output[b] if r.get("group_id")}
        if a_keys & b_keys or a_groups & b_groups:
            raise ValueError("Native prompt/context/conversation split leakage")
    return output, quarantine, dict(counts)


def protected_files(root):
    paths = list((root / "data/evaluation").glob("*.jsonl"))
    paths += [root / "data/router/clean" / f"{split}.jsonl" for split in ("validation", "test")]
    for version in ("v1", "v2"):
        base = root / "data/safety/prepared" / version
        paths += [base / task / f"{split}.jsonl" for task in ("sft", "router") for split in ("validation", "test")]
        paths += list((base / "evaluation").glob("*.jsonl"))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected holdouts missing: {missing}")
    return sorted(set(paths))


def prepare(root, raw, safety, output, sources_path, cap):
    if output.exists():
        raise ValueError("Choose a new immutable output directory")
    sources = json.loads(sources_path.read_text())
    safety_sources = json.loads((root / "configs/safety_sources.json").read_text())
    safety_sources.update(json.loads((root / "configs/safety_sources_v2.json").read_text()))
    raw_manifest = verify_inputs(raw, sources, safety)
    protected = protected_files(root)
    blocked = set()
    for path in protected:
        for row in read_jsonl(path):
            blocked.update(prompt_keys(row))
    rows = candidates(raw, safety, sources, safety_sources)
    partitions, quarantine, rejected = partition(rows, blocked, cap)
    if any(not partitions[split] for split in SPLITS):
        raise ValueError("One or more native data splits are empty")
    output.mkdir(parents=True)
    for split, values in partitions.items():
        with (output / f"{split}.jsonl").open("w") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "quarantine.json").write_text(json.dumps(quarantine, indent=2) + "\n")
    report = dict(version=1, profile="restricted_noncommercial_research", source_config=sources,
                  safety_sources={s: safety_sources[s] for s in ("pku", "esconv")}, raw_manifest=raw_manifest,
                  safety_manifest_sha256=sha(safety / "manifest.json"),
                  protected_holdouts={str(p.relative_to(root)): sha(p) for p in protected},
                  protected_prompt_count=len(blocked), input_rows=len(rows), rejected=rejected,
                  task_training_cap=cap, counts={s:dict(Counter(r["task"] for r in values)) for s,values in partitions.items()},
                  source_counts={s:dict(Counter(r["source"] for r in values)) for s,values in partitions.items()},
                  prepared_sha256={f"{s}.jsonl": sha(output / f"{s}.jsonl") for s in SPLITS},
                  prompt_split_overlap=0, context_conversation_split_overlap=0,
                  synthetic_training_examples_created=0, project_quality_review="unassessed",
                  clinical_safety_review="unassessed", production_accepted=False,
                  limitations=["Exact NFKC/case/whitespace overlap only, not semantic decontamination",
                               "Training exclusion protects existing holdouts; prior foundation pretraining overlap is unknown",
                               "Conflicting publisher splits/targets quarantine whole connected source groups",
                               "Safety eligibility follows published annotations and participant ratings, not clinical review",
                               "Dolly general responses have not had a project safety review",
                               "Science targets concatenate published correct answer and supporting passage without new reasoning",
                               "Dataset licenses restrict use and redistribution; this is not a commercial-ready training license"])
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/native/raw"))
    parser.add_argument("--safety", type=Path, default=Path("data/safety/prepared/v2"))
    parser.add_argument("--output", type=Path, default=Path("data/native/prepared/v1"))
    parser.add_argument("--sources", type=Path, default=Path("configs/native_sources.json"))
    parser.add_argument("--train-cap", type=int, default=6000)
    args = parser.parse_args()
    report = prepare(Path.cwd(), args.raw, args.safety, args.output, args.sources, args.train_cap)
    print(json.dumps({k:report[k] for k in ("counts", "source_counts", "rejected")}, indent=2))
