#!/usr/bin/env python3
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from moe_router.settings import LABELS, SAFETY_LABELS
from moe_router.data_io import read_jsonl
from moe_router.router_metrics import routing_metrics, select_threshold
from moe_router.safety import choose_expert


def metrics(actual, predicted):
    confusion = {label: Counter() for label in LABELS}
    for a, p in zip(actual, predicted): confusion[a][p] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        total = sum(confusion[label].values())
        guessed = sum(confusion[other][label] for other in LABELS)
        precision, recall = tp / max(guessed, 1), tp / max(total, 1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-10), "support": total}
    return {"accuracy": sum(a == p for a, p in zip(actual, predicted)) / max(len(actual), 1), "macro_f1": sum(v["f1"] for v in per_class.values()) / 3, "per_class": per_class, "confusion": {label: dict(counts) for label, counts in confusion.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Independent held-out JSONL with text and label")
    parser.add_argument("--output", type=Path, default=Path("outputs/router-evaluation.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibrate-validation", action="store_true", help="Report threshold diagnostics on validation.jsonl only; never modify checkpoint")
    args = parser.parse_args()
    if args.calibrate_validation and args.data.name != "validation.jsonl":
        parser.error("Threshold calibration requires validation.jsonl; test sets cannot tune thresholds")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    labels = {int(k): v.upper() for k, v in model.config.id2label.items()}
    if set(labels.values()) not in (set(LABELS), set(SAFETY_LABELS)): raise SystemExit("Model requires semantic labels")
    rows = read_jsonl(args.data)
    if not rows: raise SystemExit("Evaluation data is empty")
    if any(row.get("label") not in (*SAFETY_LABELS, "BENIGN") for row in rows): raise SystemExit("Unknown evaluation label")
    actual, predicted, routing_probabilities = [], [], []
    started = time.perf_counter()
    for offset in range(0, len(rows), 32):
        batch = rows[offset:offset + 32]
        safety_enabled = "SAFETY" in labels.values()
        window = int(getattr(model.config, "router_max_length", 128))
        kwargs = dict(padding=True, truncation=True, max_length=window, return_tensors="pt")
        if safety_enabled:
            kwargs.update(stride=window // 4, return_overflowing_tokens=True)
        encoded = tokenizer([r["text"] for r in batch], **kwargs)
        mapping = encoded.pop("overflow_to_sample_mapping", torch.arange(len(batch))).tolist()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(mapping), 32):
                inputs = {key: value[start:start + 32].to(args.device) for key, value in encoded.items()}
                chunks.append(model(**inputs).logits.softmax(-1).cpu())
        probabilities = torch.cat(chunks)
        actual.extend(r["label"] for r in batch)
        for index in range(len(batch)):
            positions = [i for i, sample in enumerate(mapping) if sample == index]
            first = probabilities[positions[0]].tolist()
            maximum = float(probabilities[positions, next(i for i, label in labels.items() if label == "SAFETY")].max()) if safety_enabled else None
            selected = choose_expert(tuple(labels[i] for i in range(len(labels))), first, maximum,
                                     float(getattr(model.config, "safety_threshold", .5)))[0]
            predicted.append(selected)
            routing_probabilities.append((first, maximum))
    report = routing_metrics(actual, predicted, SAFETY_LABELS) if "SAFETY" in actual or "BENIGN" in actual else metrics(actual, predicted)
    categories = sorted({category.strip() for row in rows for category in row.get("categories", "").split(',') if category.strip()})
    if categories:
        report["category_safety_recall"] = {}
        for category in categories:
            ids = [i for i, row in enumerate(rows) if row["label"] == "SAFETY" and category in [x.strip() for x in row.get("categories", "").split(',')]]
            if ids:
                report["category_safety_recall"][category] = dict(support=len(ids), recall=sum(predicted[i] == "SAFETY" for i in ids) / len(ids))
    report.update({"model": str(args.model), "data": str(args.data), "rows": len(rows), "seconds": time.perf_counter() - started, "independence": "Caller must establish dataset independence from model training"})
    report["source_metrics"] = {}
    for source in sorted({row.get("source", "unknown") for row in rows}):
        ids = [i for i, row in enumerate(rows) if row.get("source", "unknown") == source]
        report["source_metrics"][source] = routing_metrics([actual[i] for i in ids], [predicted[i] for i in ids], SAFETY_LABELS)
    if args.calibrate_validation:
        if "SAFETY" not in labels.values():
            parser.error("Calibration requires a four-way safety router")
        report["validation_calibration"] = select_threshold(actual, routing_probabilities, tuple(labels[i] for i in range(len(labels))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    predictions_path = args.output.with_name(args.output.stem + "-predictions.jsonl")
    with predictions_path.open("w") as stream:
        for row, guess, (first, maximum) in zip(rows, predicted, routing_probabilities):
            stream.write(json.dumps(dict(prompt_sha256=row.get("prompt_sha256"), source=row.get("source"),
                                         actual=row["label"], predicted=guess, probabilities=first,
                                         safety_max=maximum, categories=row.get("categories", ""))) + "\n")
    print(json.dumps(report, indent=2))
