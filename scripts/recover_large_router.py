#!/usr/bin/env python3
"""Strictly recover per-tensor chunks into a NEW standard checkpoint directory."""
import argparse
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer, DebertaV2ForSequenceClassification
from moe_router.settings import LABELS

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--destination", type=Path, required=True)
args = parser.parse_args()
if args.destination.exists():
    raise SystemExit("Choose a new destination; existing checkpoints are preserved")
config = AutoConfig.from_pretrained(args.source, local_files_only=True)
config.id2label = dict(enumerate(LABELS))
config.label2id = {label: i for i, label in enumerate(LABELS)}
model = DebertaV2ForSequenceClassification(config)
state = {p.stem: torch.load(p, map_location="cpu", weights_only=True) for p in args.source.glob("*.pt")}
model.load_state_dict(state, strict=True)
model.save_pretrained(args.destination, safe_serialization=True, max_shard_size="500MB")
AutoTokenizer.from_pretrained(args.source, local_files_only=True).save_pretrained(args.destination)
print("Recovered:", args.destination)
