"""Four-way router training with partial benign labels and validation calibration."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from moe_router.settings import SAFETY_LABELS
from moe_router.router_metrics import select_threshold, routing_metrics
from moe_router.safety import choose_expert
from moe_router.data_io import read_jsonl
from moe_router.router_training import windowed_partial_loss


def partial_loss(logits, targets, weights):
    log_probs = logits.log_softmax(-1)
    known = targets >= 0
    losses = torch.empty_like(targets, dtype=log_probs.dtype)
    losses[known] = -log_probs[known, targets[known]]
    losses[~known] = -log_probs[~known, :3].logsumexp(-1)
    row_weights = weights[torch.where(known, targets, 4)]
    return (losses * row_weights).sum() / row_weights.sum()


def infer(model, tokenizer, rows, device, max_length, batch_size):
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer([row["text"] for row in batch], max_length=max_length,
                                truncation=True, stride=max_length // 4, padding=True,
                                return_overflowing_tokens=True, return_tensors="pt")
            mapping = encoded.pop("overflow_to_sample_mapping").tolist()
            # Bound GPU window batch size even for long/adversarial examples.
            chunks = []
            for offset in range(0, len(mapping), batch_size):
                inputs = {k: v[offset:offset + batch_size].to(device) for k, v in encoded.items()}
                chunks.append(model(**inputs).logits.softmax(-1).cpu())
            probabilities = torch.cat(chunks)
            for index in range(len(batch)):
                positions = [i for i, sample in enumerate(mapping) if sample == index]
                values.append((probabilities[positions[0]].tolist(), float(probabilities[positions, 3].max())))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/safety/prepared/v1"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/router-earlier"))
    parser.add_argument("--output", type=Path, default=Path("outputs/router-safety-v1"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--windowed-training", action="store_true")
    parser.add_argument("--max-windows", type=int, default=8, help="Exclude entire over-budget training prompts; never truncate their evidence")
    parser.add_argument("--maximum-benign-fpr", type=float)
    parser.add_argument("--balance-safety-sources", action="store_true")
    parser.add_argument("--minimum-source-recall", type=float)
    parser.add_argument("--balance-conversation-format", action="store_true", help="Add a conversation header to a hash-selected half of real training prompts, independently of label; never invent dialogue turns")
    args = parser.parse_args()
    if args.balance_safety_sources and not args.windowed_training:
        parser.error("Source balancing requires windowed training")
    if not torch.cuda.is_available():
        parser.error("Train on an allocated NVIDIA GPU")
    if args.output.exists():
        parser.error("Choose a new output directory")
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(args.model.glob("*.safetensors"))}
    if not source_hashes:
        parser.error("Source has no safetensors checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
    # Preserve existing domain weights in explicit new order; initialize only SAFETY.
    original = {str(label).upper(): int(index) for index, label in model.config.id2label.items()}
    old = model.classifier
    classifier = torch.nn.Linear(old.in_features, 4)
    torch.nn.init.normal_(classifier.weight, std=model.config.initializer_range)
    torch.nn.init.zeros_(classifier.bias)
    with torch.no_grad():
        for index, label in enumerate(SAFETY_LABELS):
            if label not in original:
                continue
            classifier.weight[index].copy_(old.weight[original[label]])
            classifier.bias[index].copy_(old.bias[original[label]])
    model.classifier = classifier
    model.num_labels = 4
    model.config.num_labels = 4
    model.config.id2label = dict(enumerate(SAFETY_LABELS))
    model.config.label2id = {label: i for i, label in enumerate(SAFETY_LABELS)}
    model.config.router_max_length = args.max_length
    model.to(device)
    train_rows = read_jsonl(args.data / "router/train.jsonl")
    validation = read_jsonl(args.data / "router/validation.jsonl")
    format_counts = Counter()
    if args.balance_conversation_format:
        # Training-only formatting augmentation: unchanged real text/labels and
        # splits. The selector uses prompt identity, never a source or label.
        formatted = []
        for row in train_rows:
            selected = int(hashlib.sha256(("format-v3:" + row["text"]).encode()).hexdigest()[:8], 16) % 2 == 0
            if selected and not row["text"].startswith("Conversation so far:\n"):
                formatted.append({**row, "text": "Conversation so far:\nUser: " + row["text"]})
                format_counts[row["label"]] += 1
            else:
                formatted.append(row)
        train_rows = formatted
    counts = Counter(row["label"] for row in train_rows)
    safety_sources = Counter(row["source"] for row in train_rows if row["label"] == "SAFETY")
    weights = torch.tensor([len(train_rows) / (5 * counts[label]) for label in (*SAFETY_LABELS, "BENIGN")], device=device)
    if args.windowed_training:
        items, excluded = [], 0
        for row in train_rows:
            encoded = tokenizer(row["text"], max_length=args.max_length, truncation=True,
                                stride=args.max_length // 4, return_overflowing_tokens=True)
            encoded.pop("overflow_to_sample_mapping", None)
            if len(encoded["input_ids"]) > args.max_windows:
                excluded += 1
                continue
            items.append(dict(windows=[{k: v[index] for k, v in encoded.items()}
                                       for index in range(len(encoded["input_ids"]))],
                              importance=(counts["SAFETY"] / (len(safety_sources) * safety_sources[row["source"]])
                                          if args.balance_safety_sources and row["label"] == "SAFETY" else 1.0),
                              target=SAFETY_LABELS.index(row["label"]) if row["label"] != "BENIGN" else -1))
        # DeBERTa shares relative-position embeddings across layers. Reentrant
        # checkpoints can backpropagate that shared graph twice; use the modern
        # non-reentrant implementation while retaining activation savings.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        excluded = 0
        encoded = tokenizer([row["text"] for row in train_rows], max_length=args.max_length, truncation=True)
        items = [{**{k: v[index] for k, v in encoded.items()},
                  "target": SAFETY_LABELS.index(row["label"]) if row["label"] != "BENIGN" else -1}
                 for index, row in enumerate(train_rows)]
    if not items:
        raise ValueError("No retained training prompts")

    def collate(rows):
        targets = torch.tensor([row["target"] for row in rows])
        if args.windowed_training:
            flattened = [window for row in rows for window in row["windows"]]
            groups = torch.tensor([index for index, row in enumerate(rows) for _ in row["windows"]])
            inputs = tokenizer.pad(flattened, return_tensors="pt")
        else:
            groups = None
            inputs = tokenizer.pad([{k: v for k, v in row.items() if k != "target"} for row in rows], return_tensors="pt")
        importance = torch.tensor([row.get("importance", 1.0) for row in rows])
        return inputs, targets, groups, importance

    loader = DataLoader(items, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    schedule = get_linear_schedule_with_warmup(optimizer, int(total_steps * .06), total_steps)
    best, history = (-1, -1), []
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        for step, (inputs, targets, groups, importance) in enumerate(loader, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = (windowed_partial_loss(logits, groups.to(device), targets, weights, importance.to(device)) if args.windowed_training
                    else partial_loss(logits, targets, weights))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite router loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            schedule.step()
            loss_sum += float(loss.detach())
            if step % 50 == 0:
                print(json.dumps(dict(epoch=epoch + 1, step=step, steps=len(loader), loss=loss_sum / step,
                                      elapsed_seconds=time.perf_counter() - started)), flush=True)
        probabilities = infer(model, tokenizer, validation, device, args.max_length, args.batch_size)
        calibrated = select_threshold([row["label"] for row in validation], probabilities, SAFETY_LABELS,
                                      maximum_benign_fpr=args.maximum_benign_fpr,
                                      safety_groups=[row["source"] for row in validation],
                                      minimum_source_recall=args.minimum_source_recall)
        if args.balance_conversation_format:
            ordinary = [{**row, "text": "Conversation so far:\nUser: " + row["text"]}
                        for row in validation if row["label"] in ("MATH", "SCIENCE", "GENERAL")]
            formatted_probabilities = infer(model, tokenizer, ordinary, device, args.max_length, args.batch_size)
            formatted_guesses = [choose_expert(SAFETY_LABELS, first, maximum, calibrated["threshold"])[0]
                                 for first, maximum in formatted_probabilities]
            formatted_metrics = routing_metrics([row["label"] for row in ordinary], formatted_guesses, SAFETY_LABELS)
            calibrated["conversation_format_validation"] = formatted_metrics
            calibrated["conversation_format_gate_passed"] = formatted_metrics["safety_false_positive_rate"] <= 0.05
            calibrated["promotion_eligible"] = calibrated["promotion_eligible"] and calibrated["conversation_format_gate_passed"]
        record = dict(epoch=epoch + 1, train_loss=loss_sum / len(loader), calibration=calibrated)
        history.append(record)
        print(json.dumps(record), flush=True)
        score = (int(calibrated["promotion_eligible"]), calibrated["validation"]["selection_score"])
        if score > best:
            best = score
            model.config.safety_threshold = calibrated["threshold"]
            model.config.safety_validation_passed = calibrated["promotion_eligible"]
            model.save_pretrained(args.output, safe_serialization=True)
            tokenizer.save_pretrained(args.output)
        (args.output / "training_run.json").write_text(json.dumps(dict(
            source_model=str(args.model), source_weights_sha256=source_hashes,
            gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            torch_version=torch.__version__, cuda_version=torch.version.cuda,
            seed=42, labels=SAFETY_LABELS, train_rows=len(train_rows), counts=dict(counts),
            max_length=args.max_length, batch_size=args.batch_size, epochs=args.epochs,
            retained_train_rows=len(items), excluded_over_window_budget=excluded,
            windowed_training=args.windowed_training, max_windows=args.max_windows,
            maximum_benign_fpr=args.maximum_benign_fpr,
            safety_source_counts=dict(safety_sources), balance_safety_sources=args.balance_safety_sources,
            minimum_source_recall=args.minimum_source_recall,
            balance_conversation_format=args.balance_conversation_format, formatted_train_counts=dict(format_counts),
            learning_rate=args.learning_rate, data_manifest_sha256=hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
            history=history, checkpoint_selection="validation only; test never read by trainer"), indent=2) + "\n")


if __name__ == "__main__":
    main()
