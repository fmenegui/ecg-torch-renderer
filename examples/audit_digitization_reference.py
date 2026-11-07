"""Compare adapter numerics against selected original notebook definitions.

The notebook is supplied explicitly and is never executed as a notebook.
This audit loads only the named functions and records the source hash.
"""

import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.signals import decode_waveform_rows, split_rows


def audit(notebook_json, out):
    import cv2
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import resample
    from ecg_gpu.digitization.reference import HengTangDigitizer
    data = json.loads(Path(notebook_json).read_text())
    notebook = json.loads(data['blob']['source'])
    wanted = {'pixel_to_series', 'rectify_highres', 'dw_after_weighted'}
    env = {'np': np, 'torch': torch, 'F': torch.nn.functional,
           'PchipInterpolator': PchipInterpolator, 'resample': resample}
    loaded = set()
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        # Notebook shell magics are never executed. Replace them with a no-op
        # only so unrelated cells containing the wanted definitions can parse.
        source = '\n'.join((line[:len(line)-len(line.lstrip())]+'pass')
                           if line.lstrip().startswith('!') else line for line in source.splitlines())
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(notebook_json), 'exec'), env)
                loaded.add(node.name)
    if loaded != wanted:
        raise ValueError('Expected reference definitions are missing')
    rng = np.random.default_rng(1973)
    errors = []
    for gap in (False, True):
        center = rng.uniform(35, 65, size=(4, 1, 97))
        y = np.arange(100).reshape(1, 100, 1)
        p = np.exp(-.5*((y-center)/1.3)**2).astype(np.float32)
        if gap:
            p[..., 40:45] = 0
        baseline = env['pixel_to_series'](torch.from_numpy(p), [50]*4, 1250)
        expected = (50-baseline)/79
        actual, observed = decode_waveform_rows(p, zero_mv=[50]*4, pixels_per_mv=79, length=1250)
        error = float(np.max(np.abs(actual-expected)))
        if error > 2e-6:
            raise AssertionError(f'Waveform decoder differs from original: {error}')
        errors.append({'gap': gap, 'max_abs_mv': error, 'observed_fraction': float(observed.mean())})
    rows = rng.normal(size=(4, 1000))
    unfused = split_rows(rows, 100, average_ii=True, einthoven=False)
    reference = env['dw_after_weighted']({name: v.values.copy() for name, v in unfused.items()},
                                         weights={'I': 1., 'II': 1., 'III': 1.})
    fused = split_rows(rows, 100, average_ii=True, einthoven=True)
    fusion_error = max(float(np.max(np.abs(reference[name]-fused[name].values))) for name in fused)
    if fusion_error > 1e-12:
        raise AssertionError('Lead fusion differs from the original notebook')
    image = rng.integers(0, 256, size=(96, 144, 3), dtype=np.uint8)
    h = np.diag([10., 10., 1.])
    grid = np.array([[[80., 80.], [1350., 80.]], [[80., 1050.], [1350., 1050.]]])
    scale = image.shape[1]/1440
    normalized = cv2.warpPerspective(image, np.diag([scale, scale, 1.]) @ h,
                                    (int(1440*scale), int(1152*scale)), flags=cv2.INTER_CUBIC)
    expected = env['rectify_highres'](normalized, grid, scale, (int(1700*scale), int(2200*scale)))
    expected = cv2.resize(expected, (4400, 1700), interpolation=cv2.INTER_CUBIC)
    backend = object.__new__(HengTangDigitizer)
    backend.cv2, backend.geometry = cv2, 'reference'
    actual, _ = backend._rectify(image, h, grid)
    if not np.array_equal(actual, expected):
        raise AssertionError('Reference geometry differs from the original notebook')
    result = {'notebook_sha256': sha256_file(notebook_json),
        'notebook_ref': data['metadata']['ref'], 'notebook_version': data['metadata']['currentVersionNumber'],
        'waveform_decoder': errors, 'lead_fusion_max_abs_mv': fusion_error,
        'reference_geometry_bitwise_equal': True,
        'limits': 'Function-level numerical parity, not GPU/CPU network-output or private-score parity'}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook-json', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    audit(**vars(parser.parse_args()))
