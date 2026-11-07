"""Fail when pyproject, ecg_gpu.__version__ and the top CHANGELOG entry disagree.

Reads the files as text so the check runs without installing Torch.
"""
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def versions():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
    init = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
                     (ROOT / 'ecg_gpu' / '__init__.py').read_text(), re.M)
    changelog = re.search(r'^##\s+(\d+\.\d+\.\d+)', (ROOT / 'CHANGELOG.md').read_text(), re.M)
    return {
        'pyproject.toml': project,
        'ecg_gpu/__init__.py': init.group(1) if init else None,
        'CHANGELOG.md (top entry)': changelog.group(1) if changelog else None,
    }


def main():
    found = versions()
    if len(set(found.values())) != 1:
        for source, version in found.items():
            print(f'{source}: {version}')
        print('Version mismatch. Update all three together.')
        return 1
    print(f"Version {found['pyproject.toml']} is consistent.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
