"""Explicit, bounded downloads and SHA-256 validation for reference artifacts."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_lock():
    return json.loads(Path(__file__).with_name('reference-lock.json').read_text())


def _artifact_path(root, relative):
    part = PurePosixPath(relative)
    root = Path(root).resolve()
    if part.is_absolute() or '..' in part.parts or not part.parts:
        raise ValueError('Artifact path must be relative and stay inside its root')
    path = root.joinpath(*part.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Artifact path escapes its root')
    return path


def verify_assets(root, *, profile='single'):
    if profile not in ('single', 'full'):
        raise ValueError('profile must be single or full')
    lock = artifact_lock()
    for item in lock['artifacts']:
        if profile not in item.get('profiles', ('single', 'full')):
            continue
        path = _artifact_path(root, item['path'])
        if not path.is_file():
            raise FileNotFoundError(f'Missing {item["path"]}. Run ecg-digitize fetch --assets {root}')
        if path.stat().st_size != item['size'] or sha256_file(path) != item['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {item["path"]}')
    lock['lock_sha256'] = sha256_file(Path(__file__).with_name('reference-lock.json'))
    return lock


def fetch_assets(root, *, profile='single', timeout=45):
    """Download only the locked single-checkpoint profile, never during inference.

    Existing mismatched files are rejected instead of silently overwritten.
    Public upstream code and weights stay in this external directory. No
    permission to redistribute them is inferred from access to their URLs.
    """
    lock = artifact_lock()
    if profile not in ('single', 'full'):
        raise ValueError('profile must be single or full')
    for item in lock['artifacts']:
        if profile not in item.get('profiles', ('single', 'full')):
            continue
        path = _artifact_path(root, item['path'])
        if path.exists():
            if path.stat().st_size == item['size'] and sha256_file(path) == item['sha256']:
                continue
            raise ValueError(f'Refusing to overwrite mismatched artifact: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with urllib.request.urlopen(item['url'], timeout=timeout) as response:
                if item.get('format') == 'kaggle-notebook-python':
                    raw = response.read(5_000_001)
                    if len(raw) > 5_000_000:
                        raise ValueError('Notebook response exceeds size limit')
                    data = json.loads(raw)
                    source = data['blob']['source']
                    notebook = json.loads(source)
                    source = '\n\n'.join(''.join(c['source']) for c in notebook['cells']
                                         if c['cell_type'] == 'code').encode()
                    chunks = (source,)
                else:
                    chunks = iter(lambda: response.read(1024*1024), b'')
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    written = 0
                    for chunk in chunks:
                        written += len(chunk)
                        if written > item['size']:
                            raise ValueError(f'Artifact exceeds locked size: {item["path"]}')
                        stream.write(chunk)
            if written != item['size'] or sha256_file(temporary) != item['sha256']:
                raise ValueError(f'Download checksum mismatch: {item["path"]}')
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return verify_assets(root, profile=profile)
