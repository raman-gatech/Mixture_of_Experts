import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('generated_archive', Path(__file__).resolve().parents[1] / 'scripts/archive_generated_export.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class GeneratedArchiveTest(unittest.TestCase):
    def fixture(self, root):
        source, base = root / 'failed-generated', root / 'canonical'
        source.mkdir()
        base.mkdir()
        data = bytes(range(256)) * 160
        (base / 'model.safetensors').write_bytes(data)
        modified = bytearray(data)
        modified[3:12] = b'changed!!'
        (source / 'model.safetensors').write_bytes(modified)
        (source / 'config.json').write_text('{"private": true}')
        (source / 'export-checks.json').write_text('{"passed": false}')
        return source, base

    def test_actual_compression_restoration_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, base = self.fixture(root)
            before = archive.inventory(source)
            packed = root / 'archive'
            manifest = archive.archive(source, base, packed)
            self.assertEqual(before, archive.inventory(source))
            self.assertFalse(manifest['source_removed'])
            restored = root / 'restored'
            result = archive.restore(packed, restored)
            self.assertTrue(result['all_source_bytes_restored'])
            self.assertEqual(before, archive.inventory(restored))
            self.assertEqual((restored / 'model.safetensors').stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError):
                archive.restore(packed, restored)
            delta = packed / manifest['weights']['model.safetensors']['delta_file']
            original = delta.read_bytes()
            delta.write_bytes(original + b'corruption')
            output = root / 'bad-delta'
            with self.assertRaisesRegex(ValueError, 'Delta hash'):
                archive.restore(packed, output)
            self.assertFalse(output.exists())
            delta.write_bytes(original)
            (base / 'model.safetensors').write_bytes(b'x' * 40960)
            with self.assertRaisesRegex(ValueError, 'Reference shard hash'):
                archive.restore(packed, root / 'bad-base')
            self.assertEqual(before, archive.inventory(source))

    def test_successful_export_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, base = self.fixture(root)
            (source / 'export-checks.json').write_text('{"passed": true}')
            with self.assertRaisesRegex(ValueError, 'failed generated'):
                archive.archive(source, base, root / 'archive')
            (source / 'export-checks.json').write_text('{"passed": false}')
            (source / 'reference-link').symlink_to(base / 'model.safetensors')
            with self.assertRaisesRegex(ValueError, 'regular'):
                archive.archive(source, base, root / 'archive')
            self.assertFalse((root / 'archive').exists())


if __name__ == '__main__':
    unittest.main()
