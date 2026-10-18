"""Export reproducible adverse-capture cases using entirely artificial ECGs.

Run from the repository: python examples/capture_scenarios.py --out build/scenarios
File encoding is outside the renderer and outside the training hot path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import CAPTURE_SCENARIOS, ECGImageLayer, capture_scenario
from ecg_gpu.demo import demo_signals
from ecg_gpu.gpu_cli import plain


def export_scenarios(out, *, device='cpu', width=1024, height=768, severity=.8):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    values, fs, _ = demo_signals(fs=500, duration=10)
    signals = torch.tensor(values, device=device)[None]
    layouts = ['3x4+r1', '6x2', '12x1', '3x4-cabrera+r1', '3x4+r1', '6x2+r1', '3x4+r3']
    tile_w, label_h = 768, 48
    tile_h = round(height * tile_w / width)
    sheet = Image.new('RGB', (2*tile_w, 4*(tile_h+label_h)), '#e8e7e3')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    cases = []
    for index, name in enumerate(CAPTURE_SCENARIOS):
        config, parameters = capture_scenario(name, width=width, height=height, severity=severity)
        parameters['layout'] = layouts[index]
        seed = 20261018 + index
        layer = ECGImageLayer(fs=fs, config=config, parameters=parameters,
                              seed=seed, randomize=False).to(device).eval()
        batch = layer.render_batch(signals, stage='all', return_instances=True)
        array = (batch.image[0].cpu().permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(array)
        filename = name+'.png'
        image.save(out / filename)
        x, y = index % 2 * tile_w, index // 2 * (tile_h+label_h)
        draw.text((x+16, y+12), name.replace('_', ' ')+' | '+layouts[index], fill='#242b31', font=font)
        sheet.paste(image.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, y+label_h))
        cases.append({'scenario': name, 'seed': seed, 'image': filename,
                      'sha256': hashlib.sha256((out/filename).read_bytes()).hexdigest(),
                      'valid_fraction': batch.targets['valid'].mean().item(),
                      'all_12_leads_active': bool(batch.metadata['active'][0, :12].all()),
                      'configuration': layer.renderer.specification(),
                      'metadata': plain(batch.metadata)})
    draw.text((tile_w+24, 3*(tile_h+label_h)+70), 'Artificial ECG signals. Procedural captures.', fill='#242b31', font=font)
    draw.text((tile_w+24, 3*(tile_h+label_h)+108), 'No real-image equivalence benchmark completed.', fill='#242b31', font=font)
    sheet.save(out/'scenarios.jpg', quality=94, subsampling=0)
    report = {'schema': 'ecg-capture-scenarios-v1', 'device': str(device),
              'torch_version': str(torch.__version__), 'source': 'demo_signals, entirely artificial',
              'source_sha256': hashlib.sha256(np.asarray(values, dtype='<f4').tobytes()).hexdigest(),
              'source_shape': list(values.shape), 'source_dtype': '<f4', 'sample_rate_hz': fs,
              'severity': severity, 'clinical_validation': False, 'real_image_benchmark': False,
              'cases': cases}
    (out/'scenarios.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--severity', type=float, default=.8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = export_scenarios(args.out, device=args.device, width=args.width,
                              height=args.height, severity=args.severity)
    print(json.dumps({'cases': len(report['cases']), 'device': report['device'], 'out': str(args.out)}))
