import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('native_archive', Path(__file__).resolve().parents[1] / 'scripts/native_model_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class NativeArchiveTest(unittest.TestCase):
    def test_v2_requires_bound_expert_export_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=self.fixture(root)
            release=json.loads((source/'release.json').read_text())
            release.update(profile='native-internal-v2',experts_adapted=True,
                           checkpoint_sha256='checkpoint-test',adapter_scale=.5)
            proof=dict(passed=True,standard_hf_without_custom_adapter=True,
                       all_serialized_tensors_equal=True,cached_decoding_passed=True,
                       checkpoint_sha256='checkpoint-test',adapter_scale=.5)
            (source/'export-checks.json').write_text(json.dumps(proof))
            release['export_report_sha256']=archive.digest(source/'export-checks.json')
            release['metadata_sha256']['export-checks.json']=archive.digest(source/'export-checks.json')
            (source/'release.json').write_text(json.dumps(release))
            archive.pack(source,root/'packed',100000)
            self.assertTrue(archive.unpack(root/'packed',root/'restored')['passed'])
            proof['passed']=False
            (source/'export-checks.json').write_text(json.dumps(proof))
            with self.assertRaises(ValueError):
                archive.verify_release(source)

    def fixture(self, root):
        source = root / 'standard-hf-export'
        source.mkdir()
        (source / 'model.safetensors').write_bytes(bytes(range(256)) * 80)
        (source / 'config.json').write_text('{"model_type":"test_fixture"}')
        release = dict(release='test-only', profile='native-internal-v1',
                       production_accepted=False, answer_quality_accepted=False,
                       required_setting='MOE_ALLOW_SAFETY_CANDIDATE=1',
                       export_numeric_checks=dict(passed=True),
                       weights={'model.safetensors': dict(bytes=20480, sha256=archive.digest(source / 'model.safetensors'))},
                       metadata_sha256={'config.json': archive.digest(source / 'config.json')})
        (source / 'release.json').write_text(json.dumps(release))
        return source

    def test_exact_standard_export_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            before = archive.inventory(source)
            packed = root / 'persistent'
            descriptor = archive.pack(source, packed, 100000)
            result = archive.unpack(packed, root / 'node-local')
            self.assertTrue(result['passed'])
            self.assertEqual(before, archive.inventory(root / 'node-local'))
            self.assertFalse(descriptor['production_accepted'])
            self.assertEqual((root / 'node-local').stat().st_mode & 0o777, 0o700)
            self.assertEqual((root / 'node-local/model.safetensors').stat().st_mode & 0o777, 0o600)
            path = packed / 'model.tar.zst'
            path.write_bytes(path.read_bytes() + b'corrupt')
            with self.assertRaisesRegex(ValueError, 'archive changed'):
                archive.unpack(packed, root / 'corrupt')
            self.assertFalse((root / 'corrupt').exists())
            self.assertEqual(before, archive.inventory(source))

    def test_capped_compression_preserves_source_and_release_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            before = archive.inventory(source)
            with self.assertRaisesRegex(ValueError, 'quota-safe'):
                archive.pack(source, root / 'too-small', 10)
            self.assertEqual(before, archive.inventory(source))
            self.assertFalse((root / 'too-small/.model.tar.zst.partial').exists())
            self.assertFalse((root / 'too-small/archive.json').exists())
            def consumed_reserve():
                raise ValueError('Concurrent quota reserve lost')
            with self.assertRaisesRegex(ValueError, 'Concurrent quota'):
                archive.pack(source, root / 'concurrent-quota', 100000, headroom_check=consumed_reserve)
            self.assertFalse((root / 'concurrent-quota/.model.tar.zst.partial').exists())
            self.assertEqual(before, archive.inventory(source))
            release = json.loads((source / 'release.json').read_text())
            release['export_numeric_checks']['passed'] = False
            (source / 'release.json').write_text(json.dumps(release))
            with self.assertRaisesRegex(ValueError, 'Numerically checked'):
                archive.pack(source, root / 'not-qualified', 100000)

    def test_exact_known_legacy_native_release_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            release = json.loads((source / 'release.json').read_text())
            release.pop('profile')
            release['release'] = 'native-internal-v1'
            (source / 'release.json').write_text(json.dumps(release))
            descriptor = archive.pack(source, root / 'legacy', 100000)
            self.assertEqual(descriptor['profile'], 'native-internal-v1')
            archive.unpack(root / 'legacy', root / 'restored')
            self.assertEqual(archive.inventory(source), archive.inventory(root / 'restored'))
            release['release'] = 'unknown-profile'
            (source / 'release.json').write_text(json.dumps(release))
            with self.assertRaises(ValueError):
                archive.verify_release(source)

    def test_link_and_traversal_members_rejected(self):
        for name, kind in [('../outside', tarfile.REGTYPE), ('config.json', tarfile.SYMTYPE)]:
            with self.subTest(name=name, kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = self.fixture(root)
                packed = root / 'archive'
                descriptor = archive.pack(source, packed, 100000)
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode='w') as tar:
                    info = tarfile.TarInfo(name)
                    info.type = kind
                    info.linkname = '/etc/passwd' if kind == tarfile.SYMTYPE else ''
                    info.size = 0
                    tar.addfile(info)
                encoded = subprocess.run([archive.binary(), '-c'], input=stream.getvalue(), stdout=subprocess.PIPE, check=True).stdout
                (packed / 'model.tar.zst').write_bytes(encoded)
                descriptor.update(archive_sha256=archive.digest(packed / 'model.tar.zst'), archive_bytes=len(encoded))
                (packed / 'archive.json').write_text(json.dumps(descriptor))
                with self.assertRaises(ValueError):
                    archive.unpack(packed, root / 'restored')
                self.assertFalse((root / 'outside').exists())


if __name__ == '__main__':
    unittest.main()
