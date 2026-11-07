"""Generate paired ECG datasets from an explicit patient/source manifest."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from .gpu_cli import load_signals, normalized_input_fingerprint
from .gpu_module import ECGImageLayer
from .paired_dataset import export_paired_dataset, paired_samples_from_render
from .scenarios import CAPTURE_SCENARIOS, capture_scenario
from .seeding import render_seed


def generate_dataset(manifest_path, out, *, views=1, layouts=('3x4+r1', '6x2', '12x1'),
                     scenario='clean_print', width=512, height=384, severity=.6,
                     gain_mm_per_mv=10, device='cpu', seed=1973, val_fraction=.2, test_fraction=.2):
    """Read local NPY/NPZ/WFDB records and export all paired tasks.

    The manifest contains ``records`` with ``input``, ``source_id`` and
    ``group_id``. Input paths are relative to the manifest. Each input is one
    recording. Optional ``fs`` and ``units`` use the existing signal loader.
    The original waveform is not changed to manufacture overlapping traces.
    Layout/gain can change physical overlap while calibration stays aligned.
    """
    if type(views) is not int or views < 1 or not layouts:
        raise ValueError('Positive views and at least one layout required')
    if gain_mm_per_mv not in (5, 10, 20):
        raise ValueError('gain_mm_per_mv must be 5, 10 or 20')
    path = Path(manifest_path).resolve()
    source = json.loads(path.read_text(encoding='utf-8'))
    entries = source.get('records', [])
    if not entries:
        raise ValueError('Manifest must contain a nonempty records list')
    dev = torch.device(device)
    config, parameters = capture_scenario(scenario, width=width, height=height, severity=severity)
    config = replace(config, antialias_samples=2)
    parameters['gain_index'] = (5, 10, 20).index(gain_mm_per_mv)
    source_plan, sources = [], set()
    for entry in entries:
        for key in ('source_id', 'group_id', 'input'):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f'Manifest {key} must be a nonempty string')
        if entry['source_id'] in sources:
            raise ValueError('Manifest source_id must be unique. Use views for repeated captures')
        sources.add(entry['source_id'])
        values, rates, lengths = load_signals(path.parent / entry['input'], entry.get('fs'), entry.get('units'))
        if len(values) != 1:
            raise ValueError('Each manifest input must contain exactly one recording')
        length = values.shape[-1] if lengths is None else int(lengths[0])
        source_hash = normalized_input_fingerprint(values[..., :length], rates, None)['sha256']
        index = len(sources) - 1
        for view in range(views):
            source_plan.append({'source_id': entry['source_id'], 'group_id': entry['group_id'],
                'sample_id': f'record-{index:05d}-view-{view:03d}', 'provenance': {'source_sha256': source_hash}})
    layer = ECGImageLayer(config=config, parameters=parameters, randomize=False).to(dev).eval()

    def captures():
        for index, entry in enumerate(entries):
            values, rates, lengths = load_signals(path.parent / entry['input'], entry.get('fs'), entry.get('units'))
            signals = torch.from_numpy(values).to(dev)
            for view in range(views):
                capture_seed = render_seed(seed, entry['source_id'], view_id=view)
                layout = layouts[(index * views + view) % len(layouts)]
                rendered = layer.render_batch(signals, torch.as_tensor(rates, device=dev), lengths=lengths,
                    generator=torch.Generator(device=dev).manual_seed(capture_seed), parameters={'layout': layout},
                    stage='trace', return_instances=True, return_name_instances=True,
                    return_baselines=False, return_instance_boxes=False, return_field=True)
                yield from paired_samples_from_render(rendered, source_ids=[entry['source_id']],
                    group_ids=[entry['group_id']], sample_ids=[f'record-{index:05d}-view-{view:03d}'],
                    provenance=[{'seed': capture_seed, 'layout': layout, 'scenario': scenario,
                                 'configuration': layer.renderer.specification(),
                                 'source_manifest': path.name}])
    return export_paired_dataset(captures(), out, val_fraction=val_fraction,
                                 test_fraction=test_fraction, seed=seed, source_plan=source_plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--views', type=int, default=1)
    parser.add_argument('--layouts', default='3x4+r1,6x2,12x1')
    parser.add_argument('--scenario', choices=CAPTURE_SCENARIOS, default='clean_print')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--severity', type=float, default=.6)
    parser.add_argument('--gain-mm-per-mv', type=int, choices=(5, 10, 20), default=10)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--seed', type=int, default=1973)
    parser.add_argument('--val-fraction', type=float, default=.2)
    parser.add_argument('--test-fraction', type=float, default=.2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = generate_dataset(args.manifest, args.out, views=args.views,
        layouts=tuple(x.strip() for x in args.layouts.split(',') if x.strip()), scenario=args.scenario,
        width=args.width, height=args.height, severity=args.severity, gain_mm_per_mv=args.gain_mm_per_mv,
        device=args.device, seed=args.seed, val_fraction=args.val_fraction, test_fraction=args.test_fraction)
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()
