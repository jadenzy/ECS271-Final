"""Download this project's pinned public models.

    python scripts/download_models.py
    python scripts/download_models.py --verify-only
    python scripts/download_models.py --only embeddinggemma2 qwen llama_cpp

Uses configs/assets.lock.json, verifies file sizes/SHA-256, and skips valid
downloads. Interrupted downloads can be resumed by rerunning this command.
Datasets, Python packages, and NVIDIA drivers are installed separately.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from urllib.parse import urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'embeddinggemma2': 'embedding', 'qwen': 'answer', 'llama_cpp': 'llama_cpp'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def valid_file(path, entry):
    return (path.is_file() and path.stat().st_size == entry['bytes']
            and sha256(path) == entry['sha256'])


def model_path(root, relative):
    parts = PurePosixPath(relative).parts
    if (len(parts) < 3 or parts[0] != 'models'
            or any(p in ('.', '..') or ':' in p or '\\' in p for p in parts)):
        raise ValueError(f'Invalid model asset path: {relative}')
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to((root / 'models').resolve()):
        raise ValueError(f'Model asset escapes models/: {relative}')
    return path


def download_hf(root, group, source, entry, target):
    from huggingface_hub import hf_hub_download
    # Strip models/<local model directory>/ to get the repository filename.
    filename = PurePosixPath(*PurePosixPath(entry['path']).parts[2:]).as_posix()
    staging = root / '.cache/model_downloads' / group
    downloaded = Path(hf_hub_download(
        repo_id=source['repo'], revision=source['revision'], filename=filename,
        local_dir=staging))
    if not valid_file(downloaded, entry):
        raise ValueError(f'Download does not match the pinned size/SHA-256: {entry["path"]}')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    shutil.copyfile(downloaded, temporary)
    temporary.replace(target)


def extract_runtime_archives(root, source, archives):
    """Extract only pinned runtime files, rejecting unsafe ZIP paths."""
    permitted = {entry['path'] for entry in source['files']}
    for archive in archives:
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                if member.is_dir():
                    continue
                relative = 'models/llama.cpp/' + member.filename
                target = model_path(root, relative)
                if relative not in permitted:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.part')
                with stream.open(member) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                temporary.replace(target)
    for entry in source['files']:
        if not valid_file(model_path(root, entry['path']), entry):
            raise ValueError(f'Installed runtime does not match its pinned hash: {entry["path"]}')


def download_runtime(root, source):
    import requests
    archives = []
    for entry in source['archives']:
        target = root / '.cache/model_downloads/llama_cpp' / Path(urlsplit(entry['url']).path).name
        if not valid_file(target, entry):
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.part')
            with requests.get(entry['url'], stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as output:
                    for block in response.iter_content(4 * 1024 * 1024):
                        output.write(block)
            if not valid_file(temporary, entry):
                raise ValueError('Runtime archive does not match its pinned size/SHA-256')
            temporary.replace(target)
        archives.append(target)
    extract_runtime_archives(root, source, archives)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=tuple(GROUPS),
                        default=list(GROUPS), help='components to download; default: all')
    parser.add_argument('--verify-only', action='store_true', help='offline integrity check; no writes')
    args = parser.parse_args(argv)
    locked = json.loads((ROOT / 'configs/assets.lock.json').read_text(encoding='utf-8'))
    failures = 0
    for group in dict.fromkeys(args.only):
        source = locked[GROUPS[group]]
        print(f'\n{group}: {source["repo"]} @ {source["revision"]}', flush=True)
        if source.get('archives') and not args.verify_only:
            if not all(valid_file(model_path(ROOT, entry['path']), entry) for entry in source['files']):
                download_runtime(ROOT, source)
        for entry in source['files']:
            target = model_path(ROOT, entry['path'])
            if valid_file(target, entry):
                print(f'OK {entry["path"]}', flush=True)
            elif args.verify_only:
                print(f'MISSING/CHANGED {entry["path"]}', flush=True)
                failures += 1
                continue
            else:
                print(f'Download {entry["path"]} ({entry["bytes"] / 1024**2:.1f} MiB)', flush=True)
                download_hf(ROOT, group, source, entry, target)
    if failures:
        print(f'\nIntegrity check failed: {failures} missing/changed files. Run without --verify-only to repair.')
        return 1
    print('\nSelected public model assets verified. Qwen uses Q4_K_M GGUF CUDA via llama.cpp. '
          'Raw data and split manifests are copied separately.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
