"""Collect completed ICE candidates and private evaluation evidence with hashes.

This never changes the selected serving router or adapter links. Run after the
dependent evidence job completes using the user's authenticated SSH master.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/tmp/moe-ice-rswaminathan38.sock")
    parser.add_argument("--alias", default="pace")
    parser.add_argument("--models-only", action="store_true", help="Back up completed training exports while lengthy answer review generation is still running")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+", args.alias):
        parser.error("SSH alias must be a simple host/alias name")
    ssh = ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    remote_command = "cat ~/scratch/moe_router_app/outputs/safety-v2-acceptance.json"
    if args.models_only:
        remote_command = """cd ~/scratch/moe_router_app && .venv/bin/python - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('outputs')
router=json.loads((root/'router-safety-v2/training_run.json').read_text())
expert=json.loads((root/'safety-expert-v3/training_complete.json').read_text())
assert len(router['history']) == router['epochs'], 'Router training is not complete'
config=json.loads((root/'router-safety-v2/config.json').read_text())
weights=[root/'router-safety-v2/model.safetensors',root/'safety-expert-v3/adapter_model.safetensors']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(1024*1024):h.update(chunk)
 return h.hexdigest()
print(json.dumps(dict(training_complete=True,router_validation_passed=config.get('safety_validation_passed') is True,
 safety_rollout_accepted=False,independent_answer_review='unassessed',router_training=router,expert_training=expert,
 evidence_scope='Completed training and portable backups; dependent model/answer checks may still be pending',
 exported_weights={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in weights})))
PY"""
    result = subprocess.run(ssh + [args.alias, remote_command],
                            capture_output=True, text=True, check=True)
    acceptance = json.loads(result.stdout)
    if acceptance.get("training_complete") is not True:
        raise RuntimeError("Completed evidence is not available yet")
    collections = [("outputs/router-safety-v2", Path("artifacts/router-safety-v2")),
                   ("outputs/safety-expert-v3", Path("artifacts/safety_expert-v3")),
                   ("outputs/safety-answer-review-v3", Path("outputs/safety-answer-review-v3")),
                   ("outputs/safety-support-review-v3", Path("outputs/safety-support-review-v3")),
                   ("outputs/safety-api-smoke-v3", Path("outputs/safety-api-smoke-v3"))]
    if args.models_only:
        collections = collections[:2]
    for source, target in collections:
        target.mkdir(parents=True, exist_ok=True)
        if target.parts[0] == "outputs":
            target.chmod(0o700)
        subprocess.run(["rsync", "-az", "--exclude", "checkpoint-*", "--exclude", "candidate-bundle",
                        "-e", shlex.join(ssh), f"{args.alias}:scratch/moe_router_app/{source}/", str(target) + "/"], check=True)
    verified = {}
    for remote, target in (("outputs/router-safety-v2/model.safetensors", Path("artifacts/router-safety-v2/model.safetensors")),
                           ("outputs/safety-expert-v3/adapter_model.safetensors", Path("artifacts/safety_expert-v3/adapter_model.safetensors"))):
        expected = acceptance["exported_weights"][remote]
        if target.stat().st_size != expected["bytes"] or digest(target) != expected["sha256"]:
            raise RuntimeError(f"Backup hash mismatch: {target}")
        verified[str(target)] = expected
    evidence_name = "safety_improvement_v2_training.json" if args.models_only else "safety_improvement_v2_acceptance.json"
    Path("audit", evidence_name).write_text(json.dumps(acceptance, indent=2) + "\n")
    backup_record = Path("audit/safety_improvement_v2_backups.json")
    earlier = json.loads(backup_record.read_text()) if backup_record.exists() else {}
    # Keep separately verified corrective-router backups in the integrity audit.
    combined = {**earlier.get("verified_weights", {}), **verified}
    backup_record.write_text(json.dumps(dict(
        checked_at_utc=datetime.now(timezone.utc).isoformat(), verified_weights=combined,
        selected_serving_bundle_changed=False, paid_aws_resources_created=False), indent=2) + "\n")
    print(json.dumps(dict(backups_verified=True, router_validation_passed=acceptance["router_validation_passed"],
                         rollout_accepted=acceptance["safety_rollout_accepted"], independent_answer_review=acceptance["independent_answer_review"])))
