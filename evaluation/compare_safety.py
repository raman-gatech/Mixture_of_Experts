"""Blinded answer review on protected real prompts; no heuristic safety-score claim."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random

from moe_router.backend import Backend
from moe_router.data_io import read_jsonl
from moe_router.safety import SAFETY_SYSTEM
from moe_router.settings import Settings


def select_rows(rows, per_side, seed):
    rng = random.Random(seed)
    harmful = [row for row in rows if row["label"] == "SAFETY"]
    hard_benign = [row for row in rows if row["label"] == "BENIGN"]
    domains = {label: [row for row in rows if row["label"] == label] for label in ("MATH", "SCIENCE", "GENERAL")}
    rng.shuffle(harmful)
    rng.shuffle(hard_benign)
    for group in domains.values():
        rng.shuffle(group)
    # Round-robin category groups rather than allowing one dominant category.
    categories = {}
    for row in harmful:
        category = row.get("categories", "unspecified").split(',')[0].strip()
        categories.setdefault(category, []).append(row)
    selected = []
    while len(selected) < per_side and any(categories.values()):
        for group in categories.values():
            if group and len(selected) < per_side:
                selected.append(group.pop())
    # Include genuine annotated benign safety prompts, rather than allowing easy
    # domain questions to dominate the over-refusal review.
    benign = hard_benign[:per_side // 2]
    while len(benign) < per_side and any(domains.values()):
        for group in domains.values():
            if group and len(benign) < per_side:
                benign.append(group.pop())
    return selected + benign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/safety/prepared/v1/router/test.jsonl"))
    parser.add_argument("--model-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--router", type=Path, default=Path("outputs/router-safety-v1"))
    parser.add_argument("--safety-adapter", type=Path, default=Path("outputs/safety-expert-v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/safety-answer-review"))
    parser.add_argument("--per-side", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--replay-prompts", type=Path, help="Use the exact prompts from a prior answers.json; never train on them")
    parser.add_argument("--previous-safety-adapter", type=Path, help="Optional fourth arm to compare the previous SAFETY adapter with equal budgets")
    parser.add_argument("--include-routed-candidate", action="store_true", help="Evaluate actual automatic routing plus generation as another blinded arm")
    parser.add_argument("--generation-timeout", type=float, default=180)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    # Isolated candidate bundle links; original serving artifacts remain untouched.
    bundle = args.output / "candidate-bundle"
    bundle.mkdir(parents=True)
    for name in ("base", "math_expert", "science_expert", "general_expert", "combined_expert"):
        source = (args.model_root / name).resolve()
        if source.exists():
            (bundle / name).symlink_to(source, target_is_directory=True)
    (bundle / "safety_expert").symlink_to(args.safety_adapter.resolve(), target_is_directory=True)
    backend = Backend(Settings(bundle.resolve(), args.router.resolve(), (args.model_root / "base").resolve(),
                               allow_safety_candidate=True, max_new_tokens=args.max_new_tokens,
                               generation_timeout=args.generation_timeout))
    backend.load()
    modes = ["BASE_WITH_POLICY", "SAFETY_WITH_POLICY", "EXTERNAL_ROUTED"]
    if args.previous_safety_adapter:
        prior_config = json.loads((args.previous_safety_adapter / "adapter_config.json").read_text())
        current_config = json.loads((args.safety_adapter / "adapter_config.json").read_text())
        if prior_config["base_model_name_or_path"] != current_config["base_model_name_or_path"]:
            raise ValueError("Both safety adapters must declare the same base")
        backend.model.load_adapter(str(args.previous_safety_adapter.resolve()), adapter_name="SAFETY_PREVIOUS")
        modes.append("PREVIOUS_SAFETY_WITH_POLICY")
    if args.include_routed_candidate:
        modes.append("CANDIDATE_ROUTED_WITH_POLICY")
    torch = backend.torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    legacy_path = args.model_root / "router-earlier"
    legacy_tokenizer = AutoTokenizer.from_pretrained(legacy_path, local_files_only=True)
    legacy_router = AutoModelForSequenceClassification.from_pretrained(legacy_path, local_files_only=True).eval()
    if args.replay_prompts:
        corpus = {row["prompt_sha256"]: row for row in read_jsonl(args.data)}
        prior = json.loads(args.replay_prompts.read_text())
        rows = [corpus[row["prompt_sha256"]] for row in prior]
        if any(row["text"] != previous["prompt"] for row, previous in zip(rows, prior)):
            raise ValueError("Replay prompts must match original protected corpus")
    else:
        rows = select_rows(read_jsonl(args.data), args.per_side, 42)
    answers, review = [], []
    rng = random.Random(42)
    args.output.chmod(0o700)
    for index, row in enumerate(rows):
        candidates = []
        legacy_inputs = legacy_tokenizer(row["text"], max_length=128, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            legacy_index = int(legacy_router(**legacy_inputs).logits.argmax(-1).item())
        legacy_label = legacy_router.config.id2label[legacy_index].upper()
        for mode in modes:
            messages = [] if mode == "EXTERNAL_ROUTED" else [{"role": "system", "content": SAFETY_SYSTEM}]
            messages.append({"role": "user", "content": row["text"]})
            text = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = backend.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(backend.model.device)
            if inputs.input_ids.shape[-1] > backend.settings.max_input_tokens:
                candidates.append(dict(mode=mode, response="", skipped="over accepted context"))
                continue
            if mode == "CANDIDATE_ROUTED_WITH_POLICY":
                value = backend.generate(row["text"], args.max_new_tokens)
                candidates.append(dict(mode=mode, response=value["response"], generated_tokens=value["generated_tokens"],
                                       token_budget_reached=value["generated_tokens"] >= args.max_new_tokens,
                                       selected_expert=value["expert"], router_expert=value["router_expert"],
                                       routing_mode=value["routing_mode"], timings_ms=value["timings_ms"]))
                continue
            adapter = legacy_label if mode == "EXTERNAL_ROUTED" else ("SAFETY_PREVIOUS" if mode == "PREVIOUS_SAFETY_WITH_POLICY" else "SAFETY")
            backend.model.set_adapter(adapter)
            from contextlib import nullcontext
            context = backend.model.disable_adapter() if mode == "BASE_WITH_POLICY" else nullcontext()
            with context, torch.inference_mode():
                output = backend.model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                                max_time=args.generation_timeout,
                                                pad_token_id=backend.tokenizer.pad_token_id)
            generated = output[0, inputs.input_ids.shape[-1]:]
            candidates.append(dict(mode=mode, response=backend.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                                   generated_tokens=len(generated), token_budget_reached=len(generated) >= args.max_new_tokens))
        rng.shuffle(candidates)
        for slot, candidate in enumerate(candidates):
            candidate["blind_candidate"] = chr(65 + slot)
            review.append(dict(prompt_id=index, candidate=chr(65 + slot), prompt=row["text"],
                               label=row["label"], categories=row.get("categories", ""),
                               response=candidate["response"], reviewed_prompt_intent="", harmful_compliance="", over_refusal="",
                               supportive_quality="", factual_quality="", reviewer_notes=""))
        answers.append(dict(prompt_id=index, prompt_sha256=row["prompt_sha256"], label=row["label"],
                            prompt=row["text"], candidates=candidates))
        (args.output / "answers.json").write_text(json.dumps(answers, indent=2) + "\n")
        print(json.dumps(dict(completed_prompts=index + 1, total=len(rows))), flush=True)
    with (args.output / "blind_review.csv").open("w", newline="") as stream:
        # Preserve source metadata in answers.json, but do not anchor independent
        # intent/answer graders on the provider's safe/unsafe label.
        fields = [name for name in review[0] if name not in ("label", "categories")]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(review)
    (args.output / "manifest.json").write_text(json.dumps(dict(
        seed=42, source=str(args.data), source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        prompts=len(rows), modes=modes,
        safety_adapter=str(args.safety_adapter), previous_safety_adapter=str(args.previous_safety_adapter) if args.previous_safety_adapter else None,
        max_new_tokens=args.max_new_tokens, replay_prompts=str(args.replay_prompts) if args.replay_prompts else None,
        generation_timeout=args.generation_timeout,
        review_source_labels_hidden=True,
        purpose="First two modes share policy to isolate adapter effect; EXTERNAL_ROUTED reproduces the earlier router/domain-adapter baseline; review scores are deliberately blank",
        limitation="No automated claim about answer safety; this is a stratified review subset, not all holdout prompts"), indent=2) + "\n")
    backend.close()


if __name__ == "__main__":
    main()
