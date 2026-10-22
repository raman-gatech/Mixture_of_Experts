"""Create a separate, versioned research bundle without promoting candidates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from moe_router.artifacts import sha256
from moe_router.settings import SAFETY_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', type=Path, default=Path('artifacts'))
    parser.add_argument('--router', type=Path, required=True)
    parser.add_argument('--safety-adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new bundle; existing releases are preserved')
    router = args.router.resolve()
    config = json.loads((router / 'config.json').read_text())
    if set(config['id2label'].values()) != set(SAFETY_LABELS):
        raise ValueError('A four-expert semantic router is required')
    sources = {name: (args.model_root / name).resolve()
               for name in ('base', 'math_expert', 'science_expert', 'general_expert')}
    sources.update(router=router, safety_expert=args.safety_adapter.resolve())
    adapters = [json.loads((path / 'adapter_config.json').read_text())
                for name, path in sources.items() if name.endswith('_expert')]
    if len({config['base_model_name_or_path'] for config in adapters}) != 1:
        raise ValueError('All adapters must declare the same base model')
    weights = {}
    for name, source in sources.items():
        files = sorted(source.glob('*.safetensors'))
        if not files:
            raise ValueError(f'Missing model weights: {name}')
        for file in files:
            weights[f'{name}/{file.name}'] = dict(bytes=file.stat().st_size, sha256=sha256(file))
    args.output.mkdir(parents=True, mode=0o700)
    for name, source in sources.items():
        (args.output / name).symlink_to(source, target_is_directory=True)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        mode='restricted research preview', rollout_accepted=False,
        router_validation_passed=config.get('safety_validation_passed') is True,
        independent_answer_review='unassessed', weights=weights,
        sources={name:str(path) for name,path in sources.items()},
        required_setting='MOE_ALLOW_SAFETY_CANDIDATE=1',
        limitations='Routing acceptance failed; answers are not independently validated. No clinical certification.',
        portability='Symlinks refer to this ICE checkout; export with dereferencing for another host',
        internal_moe='on hold', paid_aws_resources_created=False)
    (args.output / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(bundle=str(args.output), mode=manifest['mode'], rollout_accepted=False)))


if __name__ == '__main__':
    main()
