"""Small downloader integrity checks; no network or model loading."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import download_models as downloader


def entry(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


class DownloadTests(unittest.TestCase):
    def test_paths_cannot_escape_models(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            self.assertEqual(downloader.model_path(root, 'models/model/file'),
                             root / 'models/model/file')
            for relative in ('models/../outside', '/models/model/file',
                             'models/model/C:/file', 'models/model/..\\outside'):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    downloader.model_path(root, relative)

    def test_bad_hf_download_does_not_replace_existing_model(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            downloaded, target = root / 'downloaded', root / 'model'
            downloaded.write_bytes(b'bad')
            target.write_bytes(b'existing model')
            locked = {**entry(b'good'), 'path': 'models/local/config.json'}
            with patch('huggingface_hub.hf_hub_download', return_value=str(downloaded)):
                with self.assertRaisesRegex(ValueError, 'pinned size/SHA-256'):
                    downloader.download_hf(root, 'embeddinggemma2', {'repo': 'test/model', 'revision': 'pinned'},
                                           locked, target)
            self.assertEqual(target.read_bytes(), b'existing model')

    def test_valid_hf_download_installs_pinned_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            downloaded, target = root / 'downloaded', root / 'models/local/config.json'
            downloaded.write_bytes(b'verified model asset')
            locked = {**entry(downloaded.read_bytes()), 'path': 'models/local/config.json'}
            with patch('huggingface_hub.hf_hub_download', return_value=str(downloaded)) as download:
                downloader.download_hf(root, 'qwen', {'repo': 'test/model', 'revision': 'pinned'}, locked, target)
            self.assertEqual(target.read_bytes(), downloaded.read_bytes())
            self.assertEqual(download.call_args.kwargs['revision'], 'pinned')
            self.assertEqual(download.call_args.kwargs['filename'], 'config.json')
            self.assertFalse(target.with_name('config.json.part').exists())

    def test_verify_only_reports_corruption_without_network_or_repair(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            target = root / 'models/local/config.json'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'corrupted asset')
            manifest = {'answer': {'repo': 'test/model', 'revision': 'pinned', 'files': [
                {**entry(b'correct asset'), 'path': 'models/local/config.json'}]}}
            (root / 'configs/assets.lock.json').write_text(json.dumps(manifest))
            with patch.object(downloader, 'ROOT', root), \
                    patch.object(downloader, 'download_hf', side_effect=AssertionError('Unexpected download')), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = downloader.main(['--only', 'qwen', '--verify-only'])
            self.assertEqual(result, 1)
            self.assertEqual(target.read_bytes(), b'corrupted asset')


if __name__ == '__main__':
    unittest.main()
