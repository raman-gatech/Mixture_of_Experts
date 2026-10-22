#!/usr/bin/env python3
"""Response-only QLoRA with audited filtering, dynamic padding, and resumable checkpoints."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from moe_router.safety import SAFETY_SYSTEM
from moe_router.data_io import read_jsonl


def encode_rows(path, tokenizer, max_length, safety=False, support_repeat=1):
    samples, rejected = [], Counter()
    retained_sources = Counter()
    rows = read_jsonl(path)
    for row in rows:
        human_support = (row.get("source") == "esconv" and row.get("target_eligibility") == "human_support_positive_feedback"
                         and row.get("human_written") is True and row.get("positive_feedback", 0) >= 4)
        if safety and row.get("explicitly_safe") is not True and not human_support:
            raise ValueError("Safety SFT requires explicitly safe annotated targets or provenance-backed participant-rated support")
        if not isinstance(row.get("output"), str) or not row["output"].strip():
            rejected["empty_response"] += 1
            continue
        user = row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
        messages = ([{"role": "system", "content": SAFETY_SYSTEM}] if safety else [])
        messages.append({"role": "user", "content": user})
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row["output"]}],
                                             tokenize=True, add_generation_prompt=False)
        if full[:len(prefix)] != prefix:
            raise ValueError("Chat template prefix is inconsistent")
        if len(full) > max_length:
            rejected["over_context_limit"] += 1
            continue
        if len(full) <= len(prefix):
            rejected["no_response_tokens"] += 1
            continue
        sample = dict(input_ids=full, attention_mask=[1] * len(full), labels=[-100] * len(prefix) + full[len(prefix):],
                      source=row.get("source", "unknown"))
        retained_sources[row.get("source", "unknown")] += 1
        samples.extend([sample] * (support_repeat if human_support and safety else 1))
    if not samples:
        raise ValueError(f"No valid samples in {path}")
    report = dict(input_rows=len(rows), retained_rows=len(samples), rejected=dict(rejected),
                  retained_unique_rows=sum(retained_sources.values()), retained_source_counts=dict(retained_sources),
                  source_counts=dict(Counter(row.get("source", "unknown") for row in rows)),
                  support_repeat=support_repeat,
                  limitation="Participant-rated human support is not clinical safety certification",
                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  min_tokens=min(len(row["input_ids"]) for row in samples),
                  max_tokens=max(len(row["input_ids"]) for row in samples))
    return Dataset.from_list(samples), report


class ResponseCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        labels = [row["labels"] for row in rows]
        batch = self.tokenizer.pad([{k: v for k, v in row.items() if k != "labels"} for row in rows],
                                   padding=True, pad_to_multiple_of=8, return_tensors="pt")
        length = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([row + [-100] * (length - len(row)) for row in labels])
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("artifacts/base"))
    parser.add_argument("--base-identity")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--safety", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-adapter", type=Path, help="Continue training an existing candidate without modifying its weights")
    parser.add_argument("--support-repeat", type=int, default=1, help="Training-only repetition of real supportive pairs; validation remains unique")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Train on an allocated CUDA GPU")
    if args.output.exists() and args.resume is None:
        parser.error("Choose a new output directory or resume explicitly")
    if args.support_repeat < 1:
        parser.error("support-repeat must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train, train_report = encode_rows(args.train, tokenizer, args.max_length, args.safety, args.support_repeat)
    validation, validation_report = encode_rows(args.validation, tokenizer, args.max_length, args.safety)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(base=str(args.base), base_identity=args.base_identity, train=train_report,
                      gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      validation=validation_report, safety_policy=SAFETY_SYSTEM if args.safety else None,
                      epochs=args.epochs, rank=args.rank, learning_rate=args.learning_rate,
                      max_length=args.max_length, batch_size=args.batch_size,
                      gradient_accumulation=args.gradient_accumulation, seed=42,
                      torchdynamo_disabled=os.getenv("TORCHDYNAMO_DISABLE") == "1",
                      initial_adapter=str(args.initial_adapter) if args.initial_adapter else None,
                      initial_adapter_sha256=hashlib.sha256((args.initial_adapter / "adapter_model.safetensors").read_bytes()).hexdigest() if args.initial_adapter else None,
                      checkpoint_selection="validation response loss; test not read by trainer")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(stage="tokenization_complete", train=train_report, validation=validation_report)), flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, device_map={"": 0}, torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model)
    if args.initial_adapter:
        adapter_config = json.loads((args.initial_adapter / "adapter_config.json").read_text())
        if args.base_identity != adapter_config.get("base_model_name_or_path"):
            raise ValueError("Initial adapter base identity must exactly match declared base")
        if args.rank != adapter_config["r"]:
            raise ValueError("Continuation must preserve the existing adapter rank")
        model = PeftModel.from_pretrained(model, args.initial_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=.05,
                                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                               task_type="CAUSAL_LM"))
    if args.base_identity:
        model.peft_config["default"].base_model_name_or_path = args.base_identity
    model.config.use_cache = False
    model.print_trainable_parameters()
    trainer = Trainer(model=model, train_dataset=train, eval_dataset=validation,
                      data_collator=ResponseCollator(tokenizer), args=TrainingArguments(
                          output_dir=str(args.output), num_train_epochs=args.epochs,
                          per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
                          gradient_accumulation_steps=args.gradient_accumulation,
                          learning_rate=args.learning_rate, warmup_ratio=.03, weight_decay=.01,
                          fp16=True, gradient_checkpointing=True, eval_strategy="steps", eval_steps=200,
                          save_strategy="steps", save_steps=200, save_total_limit=2,
                          load_best_model_at_end=True, metric_for_best_model="eval_loss",
                          logging_steps=10, report_to="none", seed=42, disable_tqdm=True))
    started = time.perf_counter()
    initial_validation = trainer.evaluate()
    validation_sources = {source: validation.filter(lambda row, wanted=source: row["source"] == wanted)
                          for source in validation_report["retained_source_counts"]}
    initial_by_source = {source: trainer.evaluate(eval_dataset=subset, metric_key_prefix=f"eval_{source}")
                         for source, subset in validation_sources.items()}
    print(json.dumps(dict(stage="baseline_validation", metrics=initial_validation)), flush=True)
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    final_validation = trainer.evaluate()
    final_by_source = {source: trainer.evaluate(eval_dataset=subset, metric_key_prefix=f"eval_{source}")
                       for source, subset in validation_sources.items()}
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    completed = dict(train=result.metrics, initial_validation=initial_validation, final_validation=final_validation,
                     initial_validation_by_source=initial_by_source, final_validation_by_source=final_by_source,
                     elapsed_seconds=time.perf_counter() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                     best_checkpoint=trainer.state.best_model_checkpoint,
                     limitation="Validation likelihood is not a measure of harmful-answer rate or helpfulness")
    (args.output / "training_complete.json").write_text(json.dumps(completed, indent=2) + "\n")
    print(json.dumps(completed), flush=True)


if __name__ == "__main__":
    main()
