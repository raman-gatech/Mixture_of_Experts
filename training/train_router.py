#!/usr/bin/env python3
"""Train a router on cleaned splits; choose checkpoints using validation only."""
import argparse
import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from moe_router.settings import LABELS

class Rows(Dataset):
    def __init__(self, path, tokenizer):
        self.rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.tokenizer = tokenizer
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(row["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {**{k: v.squeeze(0) for k, v in encoded.items()}, "labels": torch.tensor(LABELS.index(row["label"]))}

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=Path("data/router/clean"))
parser.add_argument("--output", type=Path, default=Path("outputs/router"))
parser.add_argument("--model", default="microsoft/deberta-v3-base")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
random.seed(args.seed); torch.manual_seed(args.seed)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
train = DataLoader(Rows(args.data / "train.jsonl", tokenizer), batch_size=16, shuffle=True)
validation = DataLoader(Rows(args.data / "validation.jsonl", tokenizer), batch_size=32)
best, history = -1, []
for epoch in range(args.epochs):
    model.train(); total = 0
    for batch in train:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(); loss = model(**batch).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step(); total += loss.item()
    model.eval(); correct = count = 0
    with torch.inference_mode():
        for batch in validation:
            batch = {k: v.to(device) for k, v in batch.items()}
            correct += (model(**batch).logits.argmax(-1) == batch["labels"]).sum().item()
            count += len(batch["labels"])
    score = correct / max(count, 1)
    history.append({"epoch": epoch + 1, "train_loss": total / max(len(train), 1), "validation_accuracy": score})
    print(json.dumps(history[-1]), flush=True)
    if score > best:
        best = score; model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
(args.output / "training_run.json").write_text(json.dumps({"seed": args.seed, "source_model": args.model, "data_manifest": json.loads((args.data / "manifest.json").read_text()), "history": history}, indent=2) + "\n")
