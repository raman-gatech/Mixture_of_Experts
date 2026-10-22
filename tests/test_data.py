import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("prepare_data", Path(__file__).resolve().parents[1] / "training/prepare_router_data.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_no_split_overlap_and_conflicts_quarantined(tmp_path):
    rows = [{"text": f"{label} question {i}", "label": label} for label in ["MATH", "SCIENCE", "GENERAL"] for i in range(30)]
    rows += [rows[0], {"text": " SCIENCE  QUESTION 0 ", "label": "GENERAL"}]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = module.prepare([source], tmp_path / "clean")
    groups = [{module.normalize(json.loads(line)["text"]) for line in (tmp_path / "clean" / f"{name}.jsonl").read_text().splitlines()} for name in ["train", "validation", "test"]]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert report["conflicting_question_groups"] == 1
    assert all("science question 0" not in group for group in groups)
