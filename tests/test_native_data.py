import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("prepare_native", Path(__file__).resolve().parents[1] / "training/prepare_native_data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def row(prompt, split="train", output="Published answer", group=None, **extra):
    return dict(instruction=prompt, input="", output=output, split=split,
                task="GENERAL", source="dolly", source_id=prompt, group_id=group, **extra)


def test_protected_holdout_excludes_connected_training_context():
    rows = [row("A protected question", group="same-passage"), row("Other question", group="same-passage"),
            row("Unrelated question")]
    output, quarantine, rejected = data.partition(rows, {data.key(" a PROTECTED  question ")})
    assert [r["instruction"] for r in output["train"]] == ["Unrelated question"]
    assert len(quarantine) == rejected["protected_existing_holdout"] == 2


def test_publisher_conflicts_quarantine_whole_conversation():
    rows = [row("First question", group="conversation"), row("Second question", "test", group="conversation")]
    output, quarantine, rejected = data.partition(rows, set())
    assert not any(output.values())
    assert rejected["publisher_or_preserved_split_conflict"] == len(quarantine) == 2


def test_targets_conflict_and_duplicates_are_not_silently_selected():
    rows = [row("Same question"), row(" same  QUESTION ", output="Contradictory answer"),
            row("Unique question"), row("Unique question")]
    output, _, rejected = data.partition(rows, set())
    assert len(output["train"]) == 1
    assert rejected["conflicting_targets"] == 2
    assert rejected["exact_duplicate"] == 1


def test_shared_context_and_instruction_components_are_split_together():
    rows = [row("First", None, group="context"), row("Second", None, group="context"),
            row(" SECOND ", None, group="different-context")]
    output, _, _ = data.partition(rows, set())
    assert sum(bool(v) for v in output.values()) == 1
    assert sum(map(len, output.values())) == 2


def test_training_caps_do_not_change_holdouts_or_drop_support_first():
    rows = [dict(row("Support"), task="SAFETY", source="esconv"),
            dict(row("Safe annotated response"), task="SAFETY", source="pku"), row("Eval", "test")]
    output, _, rejected = data.partition(rows, set(), cap=1)
    assert output["train"][0]["source"] == "esconv"
    assert len(output["test"]) == 1
    assert rejected["training_task_cap"] == 1
    with pytest.raises(ValueError):
        data.partition(rows, set(), cap=0)


def test_unicode_normalization_and_full_context_are_protected():
    item = dict(row("Ａ question"), input="Source passage\u2028with unicode")
    assert data.key("A question") in data.prompt_keys(item)
    assert data.key("A question\n\nSource passage with unicode") in data.prompt_keys(item)
    output, _, rejected = data.partition([item], {data.key("A question")})
    assert not output["train"]
    assert rejected["protected_existing_holdout"] == 1


def test_empty_targets_are_quarantined_instead_of_becoming_completions():
    output, quarantine, rejected = data.partition([row("Question", output=None)], set())
    assert not any(output.values())
    assert rejected["empty_prompt_or_response"] == len(quarantine) == 1
