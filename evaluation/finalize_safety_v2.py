"""Collect completed experiment evidence; never promote unreviewed models."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import importlib.util


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


if __name__ == "__main__":
    root = Path("outputs")
    router = load(root / "router-safety-v2/training_run.json")
    expert = load(root / "safety-expert-v3/training_complete.json")
    config = load(root / "router-safety-v2/config.json")
    api = load(root / "safety-api-smoke-v3/report.json")
    reports = {version: {split: load(root / f"safety-{version}-current-rule-{split}.json")
                         for split in ("original-test", "expanded-test", "external")} for version in ("v1", "v2")}
    spec = importlib.util.spec_from_file_location("review_summary", "evaluation/summarize_safety_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    answer_counts = {}
    for directory in ("safety-answer-review-v3", "safety-support-review-v3"):
        summary = module.build_summary(load(root / directory / "answers.json"))
        (root / directory / "unreviewed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        answer_counts[directory] = summary
    weights = [root / "router-safety-v2/model.safetensors", root / "safety-expert-v3/adapter_model.safetensors"]
    evidence = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), training_complete=True,
                    router_validation_passed=config.get("safety_validation_passed") is True,
                    independent_answer_review="unassessed", safety_rollout_accepted=False,
                    real_api_probes_passed=all(row["http_status"] == 200 and row.get("nonempty_response") for row in api["probes"]),
                    router_training=router, expert_training=expert, router_reports=reports,
                    answer_comparisons=answer_counts, api=api,
                    exported_weights={str(path): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in weights},
                    internal_moe="on hold", paid_aws_resources_created=False,
                    next="Review equal-budget answers and policy-aligned intent; back up candidate weights locally; activate only after acceptance")
    (root / "safety-v2-acceptance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(dict(training_complete=True, router_validation_passed=evidence["router_validation_passed"],
                         rollout_accepted=False, independent_answer_review="unassessed")))
