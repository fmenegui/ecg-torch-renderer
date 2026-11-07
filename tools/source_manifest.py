"""Write a SHA-256 manifest of every tracked file for a release archive.

The manifest is generated on demand rather than committed, so it cannot drift
from the tree it describes. Example:

    python tools/source_manifest.py --out build/MANIFEST.json
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'ecg-torch-renderer-source-manifest-v5'


def tracked_files():
    output = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, check=True,
                            capture_output=True).stdout
    return sorted(name for name in output.decode().split('\0') if name)


def manifest():
    version = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
                        (ROOT / 'ecg_gpu' / '__init__.py').read_text(), re.M).group(1)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    return {
        'schema': SCHEMA,
        'version': version,
        'git_commit': commit,
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in tracked_files()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=ROOT / 'build' / 'MANIFEST.json')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest(), indent=2) + '\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
