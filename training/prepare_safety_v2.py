"""Add real annotated toxic prompts and human-written support to immutable v1.

No generated examples or inferred non-toxic -> non-safety labels. Conversation
groups and every old holdout remain protected. ESConv routing labels are scoped
by dataset purpose, not independently reviewed prompt-intent annotations.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from moe_router.data_io import read_jsonl

SPLITS = ("train", "validation", "test")
STRATEGIES = {"Question", "Affirmation and Reassurance", "Reflection of feelings",
              "Restatement or Paraphrasing", "Providing Suggestions"}


def key(text):
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def supportive_pairs(conversation):
    scores = conversation.get("survey_score", {}).get("seeker", {})
    if (int(scores.get("empathy", 0)) < 4 or int(scores.get("relevance", 0)) < 4 or
            int(scores.get("final_emotion_intensity", 5)) >= int(scores.get("initial_emotion_intensity", 0))):
        return []
    dialogue = conversation["dialog"]
    result = []
    for i, turn in enumerate(dialogue[:-1]):
        following = dialogue[i + 1]
        if (not i or turn["speaker"] != "supporter" or dialogue[i - 1]["speaker"] != "seeker" or
                turn.get("annotation", {}).get("strategy") not in STRATEGIES or
                following["speaker"] != "seeker" or
                int(following.get("annotation", {}).get("feedback", 0)) < 4 or
                len(turn["content"].split()) < 10):
            continue
        history = dialogue[:i]
        # Retain context; an isolated greeting or ambiguous follow-up is not
        # presented as an independently annotated distress request.
        if sum(len(t["content"].split()) for t in history if t["speaker"] == "seeker") < 12:
            continue
        text = (history[0]["content"] if len(history) == 1 else
                "Conversation so far:\n" + "\n".join(
                    ("User: " if t["speaker"] == "seeker" else "Assistant: ") + t["content"] for t in history))
        result.append(dict(instruction=text, input="", output=turn["content"], turn_index=i,
                           strategy=turn["annotation"]["strategy"],
                           positive_feedback=int(following["annotation"]["feedback"]),
                           target_eligibility="human_support_positive_feedback",
                           human_written=True, clinical_safety_reviewed=False))
    return result


def prepare(baseline, raw, destination):
    if destination.exists():
        raise ValueError("Choose a new immutable output directory")
    rows = {task: {split: read_jsonl(baseline / task / f"{split}.jsonl") for split in SPLITS}
            for task in ("router", "sft")}
    # Never move/delete/relabel the old rows. Any new collision is quarantined.
    existing = {row["prompt_sha256"]: split for task in rows for split in SPLITS for row in rows[task][split]}
    old_router = {row["prompt_sha256"]: row["label"] for split in SPLITS for row in rows["router"][split]}
    external = read_jsonl(baseline / "evaluation/do_not_answer.jsonl")
    blocked = {key(row["text"]) for row in external}
    rejected = Counter()
    additions = []
    negative_proxy = []
    for official in ("train", "test"):
        filename = raw / "toxicchat" / f"toxic-chat_annotation_{official}.csv"
        with filename.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["human_annotation"] != "True":
                    rejected["toxicchat_not_human_annotated"] += 1
                    continue
                text = row["user_input"].strip()
                digest = key(text)
                split = "test" if official == "test" else ("validation" if int(digest[:8], 16) % 10 == 0 else "train")
                if row["toxicity"] != "1" and row["jailbreaking"] != "1":
                    rejected["non_toxic_not_a_non_safety_label"] += 1
                    if official == "test" and digest not in existing and digest not in blocked:
                        negative_proxy.append(dict(text=text, prompt_sha256=digest, source_id=row["conv_id"],
                                                   annotation="human_non_toxic; routing intent unknown"))
                    continue
                additions.append(dict(task="router", split=split, text=text, label="SAFETY", source="toxicchat",
                                      source_id=row["conv_id"], prompt_sha256=digest,
                                      categories="human_toxic" + (",human_jailbreak" if row["jailbreaking"] == "1" else ""),
                                      annotation_scope="human toxicity/jailbreak proxy, not full application policy"))
    for conversation in json.loads((raw / "esconv/ESConv.json").read_text()):
        group = hashlib.sha256(json.dumps(conversation, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        bucket = int(group[:8], 16) % 10
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        pairs = supportive_pairs(conversation)
        if not pairs:
            rejected["esconv_no_eligible_support_pair"] += 1
            continue
        hashes = {key(row["instruction"]) for row in pairs}
        if hashes & blocked or any(digest in existing and (existing[digest] != split or old_router.get(digest) not in (None, "SAFETY")) for digest in hashes):
            rejected["esconv_conversation_collision"] += 1
            continue
        for pair in pairs:
            additions.append(dict(pair, task="sft", split=split, source="esconv", conversation_sha256=group,
                                  prompt_sha256=key(pair["instruction"])))
        # One context-complete routing example per conversation prevents longer
        # conversations dominating the router's supportive prior.
        additions.append(dict(task="router", split=split, text=pairs[0]["instruction"], label="SAFETY",
                              source="esconv", categories="SUPPORTIVE_DISTRESS", conversation_sha256=group,
                              prompt_sha256=key(pairs[0]["instruction"]),
                              annotation_scope="supportive dataset purpose; positive participant feedback, not clinical review"))
    grouped = defaultdict(list)
    for row in additions:
        grouped[row["prompt_sha256"]].append(row)
    for digest, candidates in grouped.items():
        if digest in blocked or digest in existing:
            rejected["new_exact_collision_quarantined"] += len(candidates)
            continue
        if len({row["split"] for row in candidates}) != 1:
            rejected["new_split_conflict_quarantined"] += len(candidates)
            continue
        for task in rows:
            eligible = [row for row in candidates if row["task"] == task]
            if eligible:
                selected = dict(eligible[0])
                split = selected.pop("split")
                selected.pop("task")
                rows[task][split].append(selected)
                rejected["new_duplicates"] += len(eligible) - 1
    split_keys = {split: {row["prompt_sha256"] for task in rows for row in rows[task][split]} for split in SPLITS}
    conversation_groups = {split: {row["conversation_sha256"] for task in rows for row in rows[task][split]
                                  if "conversation_sha256" in row} for split in SPLITS}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_keys[a] & split_keys[b] or conversation_groups[a] & conversation_groups[b]:
            raise ValueError("Joint prompt/conversation split leakage")
    if blocked & set.union(*split_keys.values()):
        raise ValueError("External benchmark contamination")
    for task in rows:
        for split in SPLITS:
            write(destination / task / f"{split}.jsonl", rows[task][split])
    write(destination / "evaluation/do_not_answer.jsonl", external)
    # Exact original tests remain available for apples-to-apples reporting.
    for task in rows:
        write(destination / "evaluation" / f"original_{task}_test.jsonl", read_jsonl(baseline / task / "test.jsonl"))
    write(destination / "evaluation/toxicchat_non_toxic_proxy.jsonl", negative_proxy)
    for source in ("esconv", "toxicchat"):
        write(destination / "evaluation" / f"{source}_test.jsonl", [row for row in rows["router"]["test"] if row["source"] == source])
    counts = {task: {split: dict(Counter(row.get("label", "SFT") for row in values))
                     for split, values in split_rows.items()} for task, split_rows in rows.items()}
    sources = {task: {split: dict(Counter(row["source"] for row in values))
                      for split, values in split_rows.items()} for task, split_rows in rows.items()}
    manifest = dict(version=2, counts=counts, source_counts=sources, rejected=dict(rejected),
                    baseline_manifest_sha256=hashlib.sha256((baseline / "manifest.json").read_bytes()).hexdigest(),
                    sources=json.loads(Path("configs/safety_sources_v2.json").read_text()),
                    raw_manifest=json.loads((raw / "manifest.json").read_text()),
                    joint_split_overlap=0, conversation_split_overlap=0, old_rows_unchanged=True,
                    annotation_limits=["ESConv is participant-rated support, not clinician-labelled safety",
                                       "ToxicChat toxicity is a narrower proxy; non-toxic labels were NOT used as BENIGN",
                                       "Aegis interaction-label limitations and prior domain regression contamination persist"],
                    clinical_safety_review="unassessed", synthetic_training_examples_created=0)
    manifest["prepared_sha256"] = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in destination.rglob("*.jsonl")}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("data/safety/prepared/v1"))
    parser.add_argument("--raw", type=Path, default=Path("data/safety/raw-v2"))
    parser.add_argument("--output", type=Path, default=Path("data/safety/prepared/v2"))
    args = parser.parse_args()
    result = prepare(args.baseline, args.raw, args.output)
    print(json.dumps({name: result[name] for name in ("counts", "source_counts", "rejected")}, indent=2))
