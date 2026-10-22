import importlib.util
from pathlib import Path

import pytest

from moe_router.router_metrics import select_threshold
from moe_router.safety import choose_expert
from moe_router.settings import SAFETY_LABELS
from moe_router.data_io import read_jsonl

spec = importlib.util.spec_from_file_location("prepare_v2", Path(__file__).resolve().parents[1] / "training/prepare_safety_v2.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
review_spec = importlib.util.spec_from_file_location("apply_review", Path(__file__).resolve().parents[1] / "training/apply_policy_review.py")
review = importlib.util.module_from_spec(review_spec)
review_spec.loader.exec_module(review)


def test_calibrated_threshold_is_not_bypassed_by_top_safety_class():
    assert choose_expert(SAFETY_LABELS, [.25, .2, .15, .4], .4, .8)[0] == "MATH"
    assert choose_expert(SAFETY_LABELS, [.25, .2, .15, .4], .9, .8, "SCIENCE")[0] == "SAFETY"


def test_late_safety_evidence_reduces_group_loss_and_receives_gradient():
    torch = pytest.importorskip("torch")
    from moe_router.router_training import windowed_partial_loss
    logits = torch.tensor([[4., 0., 0., -4.], [0., 0., 0., 4.]], requires_grad=True)
    loss = windowed_partial_loss(logits, torch.tensor([0, 0]), torch.tensor([3]), torch.ones(5))
    assert loss < .1
    loss.backward()
    assert logits.grad[1, 3] < 0
    assert logits.grad[0].abs().sum() == 0


def test_benign_long_prompt_penalizes_safety_in_any_window():
    torch = pytest.importorskip("torch")
    from moe_router.router_training import windowed_partial_loss
    logits = torch.tensor([[4., 0., 0., -4.], [0., 0., 0., 4.]])
    loss = windowed_partial_loss(logits, torch.tensor([0, 0]), torch.tensor([-1]), torch.ones(5))
    assert loss > 2


def test_unrepresented_group_cannot_silently_train():
    torch = pytest.importorskip("torch")
    from moe_router.router_training import windowed_partial_loss
    with pytest.raises(ValueError, match="at least one window"):
        windowed_partial_loss(torch.zeros(1, 4), torch.tensor([1]), torch.tensor([3]), torch.ones(5))


def test_easy_domains_do_not_hide_benign_false_positive_gate():
    actual = ["SAFETY"] * 9 + ["BENIGN"] + ["MATH"] * 100
    values = [([.05, .05, .05, .85], .85)] * 9 + [([.05, .05, .05, .85], .85)] + [([.9, .03, .03, .04], .04)] * 100
    assert select_threshold(actual, values, SAFETY_LABELS)["promotion_eligible"]
    assert not select_threshold(actual, values, SAFETY_LABELS, maximum_benign_fpr=.05)["promotion_eligible"]


def test_source_recall_gate_prevents_pooled_recall_hiding_support_failure():
    actual = ["SAFETY"] * 100 + ["SAFETY", "BENIGN"]
    values = [([.05, .05, .05, .85], .85)] * 100 + [([.9, .03, .03, .04], .04)] * 2
    groups = ["harm"] * 100 + ["support", "negative"]
    assert not select_threshold(actual, values, SAFETY_LABELS, safety_groups=groups, minimum_source_recall=.9)["promotion_eligible"]


def conversation(strategy="Reflection of feelings", feedback="5"):
    return dict(survey_score={"seeker": dict(empathy="5", relevance="5", initial_emotion_intensity="5", final_emotion_intensity="2")},
                dialog=[dict(speaker="seeker", content="I have been struggling with many difficult feelings after losing my job recently.", annotation={}),
                        dict(speaker="supporter", content="That sounds difficult and it makes sense that you are feeling overwhelmed right now.", annotation={"strategy": strategy}),
                        dict(speaker="seeker", content="Thank you.", annotation={"feedback": feedback})])


def test_support_targets_require_positive_feedback_and_exclude_false_self_disclosure():
    pairs = data.supportive_pairs(conversation())
    assert len(pairs) == 1 and pairs[0]["human_written"]
    assert "explicitly_safe" not in pairs[0]
    assert pairs[0]["clinical_safety_reviewed"] is False
    assert data.supportive_pairs(conversation(feedback="2")) == []
    assert data.supportive_pairs(conversation(strategy="Self-disclosure")) == []


def test_expanded_real_corpus_preserves_every_old_row_and_conversation_groups():
    root = Path(__file__).resolve().parents[1] / "data/safety/prepared"
    if not (root / "v2").exists():
        pytest.skip("Research corpus is intentionally ignored in source checkout")
    groups, keys = {}, {}
    for split in ("train", "validation", "test"):
        groups[split], keys[split] = set(), set()
        for task in ("router", "sft"):
            old = read_jsonl(root / "v1" / task / f"{split}.jsonl")
            new = read_jsonl(root / "v2" / task / f"{split}.jsonl")
            assert new[:len(old)] == old
            assert not any(row.get("source") == "toxicchat" and row.get("label") == "BENIGN" for row in new)
            keys[split].update(row["prompt_sha256"] for row in new)
            groups[split].update(row["conversation_sha256"] for row in new if "conversation_sha256" in row)
    for a, b in (("train", "test"), ("train", "validation"), ("validation", "test")):
        assert not keys[a] & keys[b]
        assert not groups[a] & groups[b]


def test_blank_review_forms_cannot_create_training_labels():
    with pytest.raises(ValueError, match="No completed reviews"):
        review.accepted_reviews([dict(prompt_sha256="x", text="prompt")], [dict(prompt_sha256="x", text="prompt")])


def test_reviews_require_distinct_reviewers_and_cannot_touch_holdouts():
    row = dict(prompt_sha256="x", text="prompt", reviewer_1_id="r1", reviewer_2_id="r2",
               reviewer_1_intent="SUPPORTIVE_DISTRESS", reviewer_2_intent="SUPPORTIVE_DISTRESS",
               reviewer_1_reason="personal distress", reviewer_2_reason="asks for support")
    assert review.accepted_reviews([row], [dict(prompt_sha256="x", text="prompt")])["x"]["intent"] == "SUPPORTIVE_DISTRESS"
    with pytest.raises(ValueError, match="holdouts cannot"):
        review.accepted_reviews([row], [])
    with pytest.raises(ValueError, match="distinct attributed reviewers"):
        review.accepted_reviews([dict(row, reviewer_2_id="r1")], [dict(prompt_sha256="x", text="prompt")])
