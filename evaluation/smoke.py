#!/usr/bin/env python3
"""Real model smoke test; verifies router and selected expert, not answer quality."""
import argparse
import json
from pathlib import Path
from moe_router.backend import Backend
from moe_router.settings import Settings

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("outputs/gpu-smoke.json"))
args = parser.parse_args()
backend = Backend(Settings.from_env())
backend.load()
cases = [("MATH", "If John has 15 apples and gives away 7, how many apples remain?"), ("SCIENCE", "Why do plants need sunlight for photosynthesis?"), ("GENERAL", "How can I organize my daily study schedule?")]
results = []
for expected, prompt in cases:
    result = backend.generate(prompt, 64)
    result.update({"expected_expert": expected, "prompt": prompt})
    results.append(result)
    print(json.dumps(result), flush=True)
import torch
report = {"gpu": torch.cuda.get_device_name(), "peak_vram_bytes": torch.cuda.max_memory_allocated(), "results": results, "passed": all(r["expert"] == r["expected_expert"] == r["active_adapter"] and r["response"] for r in results)}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
if not report["passed"]:
    raise SystemExit("Router/expert smoke test failed; inspect output")
