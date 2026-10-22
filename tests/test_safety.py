import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from moe_router.api import create_app
from moe_router.backend import validate_artifacts
from moe_router.router_metrics import select_threshold, routing_metrics
from moe_router.safety import choose_expert
from moe_router.settings import Settings, SAFETY_LABELS
from moe_router.data_io import read_jsonl


def test_safety_has_priority_over_requested_math_and_base():
    for requested in ("MATH", "BASE", "COMBINED"):
        label, automatic, mode, confidence = choose_expert(SAFETY_LABELS, [.7, .1, .1, .1], .9, .5, requested)
        assert (label, automatic, mode, confidence) == ("SAFETY", "SAFETY", "safety_priority", .9)


def test_benign_manual_override_and_legacy_router_compatibility():
    assert choose_expert(SAFETY_LABELS, [.7, .1, .1, .1], .1, .5, "GENERAL") == ("GENERAL", "MATH", "selected", None)
    assert choose_expert(("GENERAL", "MATH", "SCIENCE"), [.1, .8, .1]) == ("MATH", "MATH", "automatic", .8)


def test_four_expert_router_cannot_start_without_safety_weights(tmp_path):
    settings = Settings(tmp_path, tmp_path / "router", tmp_path / "base")
    for directory in [settings.router_dir, settings.base_dir]:
        directory.mkdir()
        (directory / "model.safetensors").touch()
        (directory / "config.json").write_text("{}")
    (settings.router_dir / "config.json").write_text(json.dumps({"id2label": dict(enumerate(SAFETY_LABELS))}))
    for label in SAFETY_LABELS[:3]:
        directory = settings.expert_dir(label)
        directory.mkdir()
        (directory / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "same/base"}))
        (directory / "adapter_model.safetensors").touch()
    with pytest.raises(FileNotFoundError, match="safety_expert"):
        validate_artifacts(settings)


def test_api_reports_only_installed_experts(tmp_path):
    class BackendDouble:
        ready = False
        labels = ("MATH", "SCIENCE", "GENERAL")
        def load(self): self.ready = True
        def close(self): self.ready = False
        def generate(self, prompt, limit=None, expert_override=None): return {"expert": expert_override, "timings_ms": {"total": 1}}
    settings = Settings(tmp_path, tmp_path, tmp_path, allow_safety_candidate=True)
    backend = BackendDouble()
    with TestClient(create_app(settings, backend)) as client:
        assert client.post("/chat", json={"prompt": "q", "expert": "SAFETY"}).status_code == 422
        assert client.get("/ready").json()["research_preview"] is False
        backend.labels = SAFETY_LABELS
        assert client.get("/ready").json()["experts"] == list(SAFETY_LABELS)
        assert client.get("/ready").json()["research_preview"] is True
        assert client.post("/chat", json={"prompt": "q", "expert": "SAFETY"}).json()["expert"] == "SAFETY"


def test_failed_safety_candidate_requires_isolated_evaluation(tmp_path):
    from dataclasses import replace
    settings = Settings(tmp_path, tmp_path / "router", tmp_path / "base")
    for directory in (settings.router_dir, settings.base_dir):
        directory.mkdir()
        (directory / "model.safetensors").touch()
        (directory / "config.json").write_text("{}")
    (settings.router_dir / "config.json").write_text(json.dumps({
        "id2label": dict(enumerate(SAFETY_LABELS)), "safety_validation_passed": False}))
    for label in SAFETY_LABELS:
        directory = settings.expert_dir(label)
        directory.mkdir()
        (directory / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "same/base"}))
        (directory / "adapter_model.safetensors").touch()
    with pytest.raises(ValueError, match="validation gates"):
        validate_artifacts(settings)
    assert validate_artifacts(replace(settings, allow_safety_candidate=True)) == SAFETY_LABELS
    config_path = settings.router_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["safety_validation_passed"] = True
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="answer-review acceptance"):
        validate_artifacts(settings)
    marker = settings.expert_dir("SAFETY") / "acceptance.json"
    marker.write_text(json.dumps(dict(accepted=True, adapter_sha256="wrong model", review_evidence="reviewed cases")))
    with pytest.raises(ValueError, match="does not match these adapter weights"):
        validate_artifacts(settings)


def test_threshold_calibration_counts_benign_false_positives():
    actual = ["SAFETY", "MATH", "SCIENCE", "GENERAL", "BENIGN"]
    probabilities = [([.1, .1, .1, .7], .7), ([.8, .1, .05, .05], .05),
                     ([.1, .8, .05, .05], .05), ([.1, .05, .8, .05], .05), ([.2, .2, .4, .2], .2)]
    result = select_threshold(actual, probabilities, SAFETY_LABELS)
    assert result["promotion_eligible"]
    assert result["validation"]["safety_recall"] == 1
    assert result["validation"]["safety_false_positive_rate"] == 0


def test_hard_benign_false_positives_are_reported_separately():
    report = routing_metrics(["MATH", "SCIENCE", "GENERAL", "BENIGN", "SAFETY"],
                             ["MATH", "SCIENCE", "GENERAL", "SAFETY", "SAFETY"], SAFETY_LABELS)
    assert report["safety_false_positive_rate"] == .25
    assert report["benign_safety_false_positive_rate"] == 1
    assert report["domain_safety_false_positive_rate"] == 0


def test_positive_only_benchmark_cannot_measure_false_positives_or_calibrate():
    report = routing_metrics(["SAFETY"], ["SAFETY"], SAFETY_LABELS)
    assert report["safety_false_positive_rate"] is None
    with pytest.raises(ValueError, match="both safety and non-safety"):
        select_threshold(["SAFETY"], [([.1, .1, .1, .7], .7)], SAFETY_LABELS)


spec = importlib.util.spec_from_file_location("prepare_safety", Path(__file__).resolve().parents[1] / "training/prepare_safety_data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def test_safer_but_unsafe_response_is_never_used_as_target():
    row = dict(response_0="unsafe", response_1="safe", is_response_0_safe=False,
               is_response_1_safe=True, better_response_id=0, safer_response_id=0)
    assert data.select_safe_response(row) == ("safe", 1)
    row["is_response_1_safe"] = False
    assert data.select_safe_response(row) is None


def test_actual_prepared_safety_corpus_has_joint_disjoint_splits():
    root = Path(__file__).resolve().parents[1] / "data/safety/prepared/v1"
    if not root.exists():
        pytest.skip("Downloaded corpus is deliberately not included in source checkout")
    groups = {}
    for split in ("train", "validation", "test"):
        groups[split] = set()
        for task in ("router", "sft"):
            rows = read_jsonl(root / task / f"{split}.jsonl")
            groups[split].update(row["prompt_sha256"] for row in rows)
            if task == "sft":
                assert all(row["explicitly_safe"] is True for row in rows)
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["validation"]
    assert not groups["validation"] & groups["test"]
    external = {data.prompt_key(row["text"]) for row in read_jsonl(root / "evaluation/do_not_answer.jsonl")}
    assert not external & set.union(*groups.values())


def test_jsonl_preserves_unicode_line_separators(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"text": "before\u2028after"}, ensure_ascii=False) + "\n")
    assert read_jsonl(path) == [{"text": "before\u2028after"}]
