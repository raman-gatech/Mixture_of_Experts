import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest


def packager():
    path = Path(__file__).resolve().parents[1] / 'scripts/package_source.py'
    spec = importlib.util.spec_from_file_location('source_packager', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_archive_excludes_private_and_generated_files(tmp_path, monkeypatch):
    module = packager()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    for directory in ('src', 'artifacts', 'research', 'audit', 'src/__pycache__'):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / 'src/app.py').write_text('value = 1\n')
    (tmp_path / 'src/__pycache__/private.py').write_text('private cache')
    (tmp_path / 'src/unsafe_link.py').symlink_to(tmp_path / '.env')
    (tmp_path / '.env').write_text('MOE_API_TOKEN=private')
    (tmp_path / '.env.example').write_text('MOE_API_TOKEN=\n')
    (tmp_path / 'artifacts/model.safetensors').write_bytes(b'weights')
    (tmp_path / 'research/original.py').write_text('private research')
    (tmp_path / 'audit/private.json').write_text('{}')
    target = tmp_path / 'dist/source.zip'
    monkeypatch.setattr(sys, 'argv', ['package_source', '--output', str(target)])
    module.main()
    original = target.read_bytes()
    module.main()
    assert target.read_bytes() == original
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == {'external-moe-router/src/app.py', 'external-moe-router/.env.example'}


def test_source_archive_rejects_embedded_credentials(tmp_path, monkeypatch):
    module = packager()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/app.py').write_text('token = "hf_' + 'a' * 24 + '"\n')
    target = tmp_path / 'source.zip'
    monkeypatch.setattr(sys, 'argv', ['package_source', '--output', str(target)])
    with pytest.raises(ValueError, match='possible credential'):
        module.main()
    assert not target.exists()
