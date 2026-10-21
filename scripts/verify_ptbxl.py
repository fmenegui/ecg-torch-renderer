"""Verify original metadata and every waveform byte without extracting archives."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()

def verify(root):
    m = json.loads((root / 'manifest.json').read_text())
    assert m['records'] == 21799 and len(m['files']) == 43601
    checked = 0
    for row in m['files']:
        if row['archive'] is None:
            p = root / row['path']
            assert p.stat().st_size == row['bytes'], row['path']
            with p.open('rb') as f:
                assert digest(f) == row['sha256'], row['path']
            checked += 1
    for part in m['archives']:
        p = root / part['path']
        assert p.stat().st_size == part['bytes'], str(p)
        with p.open('rb') as f:
            assert digest(f) == part['sha256'], str(p)
        expected = {r['path']: r for r in m['files'] if r['archive'] == part['path']}
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            assert len(names) == len(set(names)) and set(names) == set(expected)
            for name in names:
                path = PurePosixPath(name)
                assert not path.is_absolute() and '..' not in path.parts
                assert z.getinfo(name).file_size == expected[name]['bytes']
                with z.open(name) as f:
                    assert digest(f) == expected[name]['sha256'], name
                checked += 1
    assert checked == len(m['files'])
    return {'status': 'verified', 'original_files': checked, 'records': m['records'], 'archives': len(m['archives'])}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
