"""Frozen image-rectification adapters. No waveform extraction is invoked.

npj normally dewarps its signal probability, not the returned RGB. Applying its
same estimated warp to RGB is explicitly an adapter output, not published RGB.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.reference import HengTangDigitizer
from ecg_gpu.digitization.runner import _write_json
from examples.digitization_npj_probe import load_upstream


class NPJRectifier:
    def __init__(self, source, device='cuda'):
        self.model, _, self.commit = load_upstream(source)
        self.device = device
        self.model.device = device
        self.model.segmentation_model.to(device)

    def rectify_image(self, rgb):
        model = self.model
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(self.device)
        torch.manual_seed(20261021)
        with torch.no_grad():
            image = model._resample_image(model.min_max_normalize(image))
            signal, grid, text = model._get_feature_maps(image)
            params = model.perspective_detector(grid)
            corners = model.cropper(signal, params)
            aligned, _, aligned_grid, _ = model._align_feature_maps(image, signal, grid, text, corners)
            mx, my = model.pixel_size_finder(aligned_grid)
        ppm = (1/mx+1/my)/2
        # The upstream optimizer requires enabled gradients for its positions.
        model.dewarper.fit(aligned_grid.squeeze(), ppm)
        with torch.no_grad():
            sampling = model.dewarper.grid[None].to(aligned.device)
            warped = F.grid_sample(aligned, sampling, align_corners=False)
            support = F.grid_sample(torch.ones_like(aligned[:, :1]), sampling, align_corners=False) >= .999
            warped = torch.where(support, warped, torch.ones_like(warped))
        return {'rectified': (warped[0].permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8),
                'aligned': (aligned[0].permute(1, 2, 0).cpu().numpy()*255).round().clip(0, 255).astype(np.uint8),
                'points_aligned_xy': np.asarray(model.dewarper.final_local_maxima)[:, ::-1],
                'support': support[0, 0].cpu().numpy(),
                'sampling_normalized': sampling[0].cpu().numpy(),
                'upstream_commit': self.commit, 'nonlinear_dewarping': True,
                'rgb_warp_is_adapter': True, 'waveform_extraction_invoked': False}


def main(images, out, source, assets, device='cuda'):
    """Run identical, ordered image inputs and retain every per-method failure."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(images.read_text())
    reports = []
    for method in ('kaggle', 'npj'):
        try:
            model = (HengTangDigitizer(assets, device=device, rectification_only=True)
                     if method == 'kaggle' else NPJRectifier(source, device=device))
        except Exception as exc:
            reports.extend({'method': method, 'id': row['id'], 'status': 'failed',
                            'phase': 'initialization', 'error': str(exc)} for row in manifest)
            _write_json(out/'references.json', reports)
            continue
        for row in manifest:
            path = images.parent/row['file']
            record = {'method': method, 'id': row['id'], 'input_sha256': sha256_file(path), 'status': 'failed'}
            try:
                if record['input_sha256'] != row['sha256']:
                    raise ValueError('Input bytes differ from locked image list')
                rgb = np.asarray(Image.open(path).convert('RGB'))
                result = model.rectify_image(rgb)
                directory = out/method/row['id']
                directory.mkdir(parents=True)
                Image.fromarray(result['rectified']).save(directory/'rectified.png')
                geometry = {k: v for k, v in result.items() if isinstance(v, np.ndarray) and k != 'rectified'}
                np.savez_compressed(directory/'geometry.npz', **geometry)
                record.update(status='returned', image=str((directory/'rectified.png').relative_to(out)),
                              waveform_extraction_invoked=False)
            except Exception as exc:
                record['error'] = type(exc).__name__+': '+str(exc)
            reports.append(record)
            _write_json(out/'references.json', reports)
            print(json.dumps(record), flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('images', 'out', 'source', 'assets'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    torch.set_num_threads(4)
    main(**vars(parser.parse_args()))
