"""RGB-only inference with a diverse-study multitask checkpoint.

The optional restored image is a diagnostic. The geometric output always
samples the original RGB, with explicit partial support and no extrapolation.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.denoising_grid import DenoisingGridNet
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.local_cells import locally_supported_cells
from ecg_gpu.digitization.support import cell_support_summary
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.grid_point_learning import decode_grid_points
from examples.digitization_grid_rectification import grid_remap


def rectify(checkpoint, image, out, device='cpu', pixels_per_mm=8.):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory')
    if not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError('Positive finite output density required')
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['model_class'] != 'DenoisingGridNet':
        raise ValueError('A diverse-study checkpoint is required')
    model = DenoisingGridNet(width=state['width']).to(device).eval()
    model.load_state_dict(state['state_dict'])
    rgb = np.asarray(Image.open(image).convert('RGB'))
    h, w = rgb.shape[:2]
    ratio = min(1024/w, 704/h)
    rw, rh = max(1, round(w*ratio)), max(1, round(h*ratio))
    canvas = np.full((704, 1024, 3), 255, np.uint8)
    canvas[:rh, :rw] = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BILINEAR))
    tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1)[None].float().to(device)/255
    with torch.inference_mode():
        output = model(tensor)
        prediction = decode_grid_points(output['heat'].sigmoid(), output['offsets'],
            threshold=float(state['threshold']), max_points=16384)
        keep = prediction['valid'][0]
        xy, scores = prediction['xy'][0, keep].cpu().numpy(), prediction['scores'][0, keep].cpu().numpy()
        clean = output['clean'][0].cpu().permute(1, 2, 0).numpy()[:rh, :rw]
    inside = ((xy >= 0) & (xy <= [rw-1, rh-1])).all(1)
    xy, scores = (xy[inside]+.5)/[rw/w, rh/h]-.5, scores[inside]
    native = organize_lattice(xy, scores)
    local = locally_supported_cells(xy, native)
    report = {'model_class': state['model_class'], 'joint_training': state['joint'],
        'checkpoint_sha256': sha256_file(checkpoint), 'input_sha256': sha256_file(image),
        'device': device, 'point_count': len(xy), 'threshold': state['threshold'],
        'native_status': native['status'], 'local_status': local['partial_status'],
        'native_cycle_conflict_fraction': native.get('conflict_fraction'),
        'local_cells': len(local['cells']), 'removed_cells': local.get('removed_cells', 0),
        'status': 'rejected', 'assumed_major_grid_mm': 5., 'pixels_per_mm': pixels_per_mm,
        'source_for_rectification': 'original RGB', 'gain_inferred': False, 'speed_inferred': False,
        'global_injectivity_checked': False}
    report['cell_support'] = cell_support_summary(local)
    np.savez_compressed(out/'points.npz', xy=xy, scores=scores,
                        selected=np.asarray(local['selected_indices']), ids=np.asarray(local['ids']), cells=np.asarray(local['cells']))
    if state['joint']:
        restored = Image.fromarray((clean*255).round().clip(0, 255).astype(np.uint8))
        restored.resize((w, h), Image.Resampling.BILINEAR).save(out/'restored-diagnostic.png')
    if local['partial_status'] == 'partial':
        ids = np.asarray(local['ids'])
        canonical = ids[:, ::-1]*(5*pixels_per_mm)
        size = (canonical.max(0)+1).astype(int)[::-1]
        if max(size) <= 4096:
            warped, support = grid_remap(rgb, xy[local['selected_indices']], canonical, ids,
                                        tuple(size), allowed_cells=local['cells'])
            Image.fromarray(warped).save(out/'rectified-original-rgb.png')
            Image.fromarray(support.astype(np.uint8)*255).save(out/'output-support.png')
            report.update(status='partial-rectification', supported_output_canvas_fraction=float(support.mean()))
        else:
            report['error'] = 'Output extent exceeds 4096-pixel diagnostic bound'
    _write_json(out/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'image', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--pixels-per-mm', type=float, default=8.)
    torch.set_num_threads(4)
    print(json.dumps(rectify(**vars(parser.parse_args()))))
