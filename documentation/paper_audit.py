"""Audit material strain and inverse accuracy against a denser Newton solve."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGRenderer, RenderConfig, __version__
from ecg_gpu.gpu_document import SURFACES


def audit(device):
    torch.set_num_threads(2)
    config = RenderConfig(height=384, width=512, profile='photo', severity=.8,
                          surface_resolution=96, antialias_samples=2, shadow_samples=8)
    coarse = ECGRenderer(config).to(device)
    dense = ECGRenderer(replace(config, geometry_solver='dense', inverse_iterations=12)).to(device)
    p = coarse.sample_parameters(8, generator=torch.Generator(device=device).manual_seed(7103))
    p['surface_mode'] = torch.arange(8, device=device)
    p['surface_strength'].fill_(.9)
    p['relief_view'][:] = torch.tensor([.38, -.25], device=device)
    p['angle'] = torch.linspace(-.12, .14, 8, device=device)
    p['quarter'].zero_()
    p['zoom'].fill_(.85)
    p['curl_x'].zero_()
    p['curl_y'].zero_()
    base = coarse._field(p)
    a = coarse.document.deform(base, p, torch.Generator(device=device).manual_seed(13))
    b = dense.document.deform(base, p, torch.Generator(device=device).manual_seed(13))
    visible = ((a['field'].abs() < .96).all(-1) & (b['field'].abs() < .96).all(-1)
               & a['solver_valid'][:, 0].bool() & b['solver_valid'][:, 0].bool())
    error = (a['field'] - b['field']).abs() * torch.tensor([512/2, 384/2], device=device)
    _, _, metric = coarse.document.surface(p, torch.Generator(device=device).manual_seed(13))
    rows = []
    for mode, name in enumerate(SURFACES):
        mask = visible[mode]
        row = {'surface': name, 'mesh_mean_abs_metric_error': float(metric[mode].abs().mean()),
               'mesh_p99_abs_metric_error': float(torch.quantile(metric[mode].abs().flatten(), .99)),
               'max_uv_error_pixels_vs_dense': float(error[mode][mask].max()),
               'max_projection_residual_pixels': float(a['projection_residual'][mode, 0][mask].max()*512/2),
               'interior_visible_fraction': float(mask.float().mean())}
        if row['max_uv_error_pixels_vs_dense'] > .2 or row['mesh_mean_abs_metric_error'] > .004:
            raise AssertionError(f'Paper geometry regression: {row}')
        rows.append(row)
    root = Path(__file__).resolve().parents[1]
    return {'schema':'ecg-material-paper-audit-v1', 'version':__version__, 'device':str(device),
            'torch_version':str(torch.__version__), 'cuda_measured':device.type == 'cuda',
            'configuration':asdict(config), 'dense_reference_iterations':12,
            'seed':7103, 'detail_seed':13, 'real_photograph_validation':False,
            'strain_definition':'|du|-1, |dv|-1 and dot(du,dv), material coordinates',
            'source_sha256':{name:hashlib.sha256((root/'ecg_gpu'/name).read_bytes()).hexdigest()
                             for name in ['gpu_paper.py','gpu_document.py','gpu_renderer.py']}, 'surfaces':rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    with torch.no_grad():
        result = audit(torch.device(args.device))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
