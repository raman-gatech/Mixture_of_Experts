"""Bind the frozen external release to actual ICE paths without moving weights."""
import json
from pathlib import Path
from moe_router.artifacts import sha256, inspect_safetensors

root=Path('/path/to/ice-home/scratch/moe_router_app')
frozen=json.loads(Path('audit/releases/external-v1.json').read_text())
actual={}
for relative,expected in frozen['weights'].items():
    candidates=[root/'artifacts'/relative, root/'outputs'/relative]
    if relative.startswith('safety_expert-v3/'):
        candidates.append(root/'outputs/safety-expert-v3'/Path(relative).name)
    if relative.startswith('router-safety-v3/'):
        candidates.append(root/'outputs/router-safety-v3'/Path(relative).name)
    match=None
    for path in candidates:
        if not path.is_file(): continue
        integrity=inspect_safetensors(path)
        if sha256(path)==expected['sha256']:
            match=dict(path=str(path.resolve()),sha256=expected['sha256'],bytes=integrity['bytes'])
            break
    if match is None: raise FileNotFoundError(f'Frozen external artifact not verified on ICE: {relative}')
    actual[relative]=match
snapshot=Path(frozen['source_snapshot']['path'])
assert sha256(snapshot)==frozen['source_snapshot']['sha256']
report=dict(release='external-v1',weights=actual,verified_weight_files=len(actual),
    source_snapshot=frozen['source_snapshot'],files_moved=False,files_overwritten=False,
    production_accepted=False,restart_checkout=str(root),restart='scripts/ice_serve_safety_preview.sbatch')
Path('audit/releases/external-v1-ice.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(external_release_verified=True,weights=len(actual))),flush=True)
