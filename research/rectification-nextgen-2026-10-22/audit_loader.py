"""Load only the audited B0 geometry, without importing optional package paths."""
import hashlib
import importlib
from pathlib import Path
import sys
import types

DEFAULT_GEOMETRY = Path(__file__).resolve().parents[2]/'ecg_gpu/digitization'
PINNED_BLOBS = {
    'complete_rectifier.py': 'b6410d473d8fb91b083b7dbdd8ae3600ee83b67b',
    'frame_mesh.py': 'e67d39932da04dcc848728abd14782fa456c8388',
    'cell_mesh.py': '6c3d29b89820bb5412704505468425f405a8b4ba',
}


def load_geometry(directory=DEFAULT_GEOMETRY):
    directory = Path(directory).resolve()
    for name, expected in PINNED_BLOBS.items():
        content = (directory/name).read_bytes()
        digest = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if digest != expected:
            raise ValueError(f'{name} is not the audited B0 Git blob. Use a checkout at the recorded commit.')
    package_name = '_ecg_swarm_audited_geometry'
    package = sys.modules.get(package_name)
    if package is not None and package.__path__ != [str(directory)]:
        raise RuntimeError('Use a fresh process for a different audited geometry directory')
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]
        sys.modules[package_name] = package
    return importlib.import_module(package_name+'.complete_rectifier')
