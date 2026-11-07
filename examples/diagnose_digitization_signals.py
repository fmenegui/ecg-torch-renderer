"""Conditional E5/E6 signal diagnostics on the frozen synthetic M3 panel.

Known print metadata and reference geometry are supplied deliberately. This
does not measure autonomous layout/calibration recognition or a native npj
pipeline. The unchanged pinned npj segmentation network is a common frontend,
followed by the existing missing-aware region decoder. No waveform gap is filled.
The dense reference field is inverted through an explicit coarse cell mesh,
whose approximation error and support are measured separately for every case.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu import ECGImageLayer, capture_scenario
from ecg_gpu.digitization.cell_mesh import CellMap, cell_triangles
from ecg_gpu.digitization.data import read_wfdb16
from ecg_gpu.digitization.regions import TraceCalibration, decode_trace_region
from ecg_gpu.schema_v05 import LEADS, LAYOUTS
from ecg_gpu.scenarios import CAPTURE_SCENARIOS
from examples.diverse_grid_study import clean_parameters, uint8
from examples.diagnose_digitization_stages import digest, write_json

WEIGHT_SHA = '17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b'
SOURCE_SHA = '549a06ed70eb1bc04a833c3fc0ec71f3cb46ba3e9fe6ad08262fd9dcd87b55a2'


def reconstruct_document(record, case):
    """Replay print parameters with original seed, never rerender a new capture."""
    index, view = case['source_index'], case['view']
    seed = 20261022+index*1009+view*47
    rng = np.random.default_rng(seed)
    offline = view in (2, 6)
    scenario = CAPTURE_SCENARIOS[(index*5+view*3) % len(CAPTURE_SCENARIOS)]
    layout = LAYOUTS[1:][(index+view*7) % (len(LAYOUTS)-1)]
    assert layout == case['layout']
    severity = float(rng.uniform(.15, .85))
    width, height = (1536, 1056) if offline else (1024, 704)
    config, p = capture_scenario('clean_print' if offline else scenario,
                               width=width, height=height, severity=severity)
    config = replace(config, max_font_size=48. if offline else 32.)
    p.update(layout=layout, speed_index=int(rng.integers(3)), gain_index=int(rng.integers(3)),
             font=int(rng.integers(2)), font_size=float(rng.uniform(16, 24))*(1.5 if offline else 1),
             grid_phase_x=float(rng.uniform(0, 5)), grid_phase_y=float(rng.uniform(0, 5)),
             grid_strength=float(rng.uniform(.18, .65)), minor_strength=float(rng.uniform(.15, .5)),
             grid_style=0, line_width=float(rng.uniform(.7, 1.6)),
             header_present=float(rng.random() > .15), label_below=bool(rng.random() > .7))
    if offline:
        p.update(paper_age=float(rng.uniform(0, .2)), coffee_strength=float(rng.uniform(0, .2)))
    values = torch.tensor(np.stack([record[n].values for n in LEADS]), dtype=torch.float32)[None]
    layer = ECGImageLayer(fs=100., config=config, parameters=p, seed=seed,
                          randomize=False, photo=not offline).eval()
    with torch.no_grad():
        original = layer.render_batch(values, photo=not offline, stage='image', as_texture=offline)
        clean_p = clean_parameters(original.metadata['parameters'])
        clean_config, _ = capture_scenario('clean_print', width=width, height=height, severity=0.)
        clean_config = replace(clean_config, max_font_size=48. if offline else 32.)
        clean_layer = ECGImageLayer(fs=100., config=clean_config, seed=seed,
                                    randomize=False, photo=False).eval()
        canonical = clean_layer.render_batch(values, parameters=clean_p,
                                             photo=False, stage='image', as_texture=offline)
    metadata = {k: v[0].detach().cpu().numpy() for k, v in canonical.metadata.items()
                if isinstance(v, torch.Tensor)}
    phase = np.array([float(clean_p[k].reshape(-1)[0]) % 5
                      for k in ('grid_phase_x', 'grid_phase_y')])
    image = uint8(canonical.image)
    scale = 1024/width
    if scale != 1:
        image = np.asarray(Image.fromarray(image).resize((1024, 704), Image.Resampling.BILINEAR))
    return image, metadata, phase, scale


def reference_field_mesh(field, valid, step=8):
    """Only adjacent observed source cells, no filling across explicit holes."""
    height, width = valid.shape
    xs = np.unique(np.r_[np.arange(0, width, step), width-1])
    ys = np.unique(np.r_[np.arange(0, height, step), height-1])
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    ri, ci = np.meshgrid(np.arange(len(ys)), np.arange(len(xs)), indexing='ij')
    source = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    physical = field[yy.ravel(), xx.ravel()].astype(float)
    ids = np.column_stack((ri.ravel(), ci.ravel()))
    keep = (valid[yy.ravel(), xx.ravel()] >= .99) & np.isfinite(physical).all(1)
    source, physical, ids = source[keep], physical[keep], ids[keep]
    triangles = cell_triangles(source, physical, ids)
    # Check the entire original 8x8 block: opaque interiors must not be bridged.
    if len(triangles):
        tri_source = source[triangles]
        safe = []
        for vertices in tri_source:
            lo, hi = vertices.min(0).astype(int), vertices.max(0).astype(int)
            safe.append(bool((valid[lo[1]:hi[1]+1, lo[0]:hi[0]+1] >= .99).all()))
        triangles = triangles[np.asarray(safe, bool)]
    return CellMap(physical, source, triangles), CellMap(source, physical, triangles), len(triangles)


def sample_reference_rgb(image, inverse, metadata, phase, scale, subgrid=1):
    """Bilinear source reconstruction, fixed declared output quadrature."""
    height, width = 704, 1024
    result = np.full((height, width, 3), 255., np.float32)
    support = np.zeros((height, width), float)
    ppm = float(metadata['ppm'])
    for start in range(0, height, 32):
        yy, xx = np.mgrid[start:min(start+32, height), :width]
        rgb_sum = np.zeros((*xx.shape, 3), float)
        count = np.zeros(xx.shape, float)
        for dy in (np.arange(subgrid)+.5)/subgrid-.5:
            for dx in (np.arange(subgrid)+.5)/subgrid-.5:
                original = (np.column_stack((xx.ravel()+dx, yy.ravel()+dy))+.5)/scale-.5
                physical = original/ppm-phase
                source = inverse(physical)
                finite = np.isfinite(source).all(1)
                inside = (finite & (source >= 0).all(1)
                          & (source <= [image.shape[1]-1, image.shape[0]-1]).all(1))
                coordinates = np.where(inside[:, None], source, -100.).astype(np.float32)
                rgb_sum += cv2.remap(image, coordinates[:, 0].reshape(xx.shape),
                    coordinates[:, 1].reshape(xx.shape), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
                count += inside.reshape(xx.shape)
        result[start:start+len(yy)] = rgb_sum/(subgrid*subgrid)
        support[start:start+len(yy)] = count/(subgrid*subgrid)
    return result.round().clip(0, 255).astype(np.uint8), support


def segmentation_model(root):
    source = root/'src/model/unet.py'
    weights = root/'weights/unet_weights_07072025.pt'
    if digest(source) != SOURCE_SHA or digest(weights) != WEIGHT_SHA:
        raise ValueError('Pinned common segmentation source/weight mismatch')
    spec = importlib.util.spec_from_file_location('m5_pinned_npj_unet', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.UNet(num_in_channels=3, num_out_channels=4,
        dims=[32, 64, 128, 256, 320, 320, 320, 320], depth=2).eval()
    state = torch.load(weights, weights_only=True, map_location='cpu')
    if isinstance(state, tuple):
        state = state[0]
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()})
    return model


@torch.inference_mode()
def trace_probability(model, image):
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float()
    lo, hi = tensor.min(), tensor.max()
    if hi == lo:
        return np.zeros(image.shape[:2], np.float32)
    probability = model((tensor-lo)/(hi-lo)).softmax(1)[:, 2]
    probability = (probability-probability.mean()).clamp_min(0)
    probability = probability/(probability.max()+1e-9)
    return probability[0].numpy()


def at_reference_times(observation, times):
    source = observation.start_s+np.arange(len(observation.values_mv))/observation.fs
    result = np.full(len(times), np.nan)
    observed = np.flatnonzero(observation.observed)
    for run in np.split(observed, np.flatnonzero(np.diff(observed) != 1)+1):
        if len(run) > 1:
            keep = (times >= source[run[0]]) & (times <= source[run[-1]])
            result[keep] = np.interp(times[keep], source[run], observation.values_mv[run])
        elif len(run) == 1:
            keep = np.isclose(times, source[run[0]], atol=1e-10, rtol=0)
            result[keep] = observation.values_mv[run[0]]
    return result


def extract(probability, support, metadata, scale, records):
    output, samples = [], {}
    ppm = float(metadata['ppm'])*scale
    speed, gain = float(metadata['speed']), float(metadata['gain'])
    half = max(4., .44*float(metadata['cell_height'])*scale)
    for i in np.flatnonzero(metadata['active'] > .5):
        lead = LEADS[int(metadata['lead_id'][i])]
        baseline = (float(metadata['baseline'][i])+.5)*scale-.5
        x0 = (float(metadata['x0'][i])+.5)*scale-.5
        x1 = (float(metadata['x1'][i])+.5)*scale-.5
        xa, xb = max(0, int(np.ceil(x0))), min(probability.shape[1], int(np.floor(x1))+1)
        ya, yb = max(0, int(np.floor(baseline-half))), min(probability.shape[0], int(np.ceil(baseline+half))+1)
        start, end = float(metadata['start_s'][i]), float(metadata['end_s'][i])
        times = np.arange(len(records[lead].values))/records[lead].fs
        times = times[(times >= start) & (times < end)]
        reference = np.interp(times, np.arange(len(records[lead].values))/records[lead].fs, records[lead].values)
        values = np.full(len(times), np.nan)
        if yb-ya >= 3 and xb > xa:
            p = probability[ya:yb, xa:xb].copy()
            # Only absent source support is masked, never grid confidence.
            p[support[ya:yb, xa:xb] < .999] = 0
            cal = TraceCalibration(str(i), lead, ppm, ppm, speed, gain, baseline-ya,
                start+(xa-x0)/(ppm*speed), int(metadata['amplitude_sign'][i]))
            observation = decode_trace_region(p, cal, confidence_threshold=.15, ambiguity_ratio=.95)
            values = at_reference_times(observation, times)
        keep = np.isfinite(values)
        mse = float(np.mean((values[keep]-reference[keep])**2)) if keep.any() else None
        output.append({'instance': int(i), 'lead': lead, 'expected_samples': len(times),
            'observed_samples': int(keep.sum()), 'mse_mv2': mse})
        samples[str(i)] = {'times': times, 'reference': reference, 'prediction': values}
    return output, samples


def run(bank, comparison, sources, upstream, out, limit=None):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    selected = [r for r in json.loads((comparison/'schedule.json').read_text())['cases'] if r['split'] == 'test']
    if limit is not None:
        selected = selected[:limit]
    source_rows = {r['source_id']: r for r in json.loads((bank/'source-records.json').read_text())}
    protocol = {'schema': 'ECG-conditional-signal-diagnostic-v1', 'date': '2026-10-26',
        'cases': [r['key'] for r in selected], 'development_only': True,
        'known_information': ['print layout', 'lead labels', 'gain', 'speed', 'baselines', 'time windows', 'reference geometry'],
        'scope': 'Conditional extraction, not autonomous ECG digitization or official npj pipeline.',
        'reference_field_mesh_step_px': 8, 'output_size': [704, 1024],
        'methods': ['E5-reference-warp-bilinear', 'E5-reference-warp-2x2', 'E6-regenerated-canonical'],
        'common_extractor': {'segmentation_weight_sha256': WEIGHT_SHA,
            'segmentation_source_sha256': SOURCE_SHA, 'signal_class': 2,
            'threshold': .15, 'ambiguity_ratio': .95, 'roi_half_cell_height': .44,
            'no_gap_filling': True, 'no_time_DC_gain_alignment': True},
        'limitations': ['E5 inverse is a validated piecewise-affine approximation of the saved dense field, not exact analytic geometry.',
            'E5/E6 difference includes capture damage, visibility, resampling and extractor transfer.',
            'E6 is a new canonical rendering with replayed print parameters, not original measured camera truth.',
            'Fixed 2x2 quadrature is an experimental arm, not a preservation guarantee.'],
        'code_sha256': digest(__file__)}
    write_json(out/'protocol.json', protocol)
    model = segmentation_model(upstream)
    all_rows = []
    for index, case in enumerate(selected):
        key = case['key']
        start_time = time.perf_counter()
        case_out = out/'cases'/key
        case_out.mkdir(parents=True, exist_ok=True)
        source = source_rows[case['source_id']]
        header = sources/(Path(source['filename_lr']).name+'.hea')
        assert digest(header) == source['header_sha256']
        assert digest(header.with_suffix('.dat')) == source['dat_sha256']
        record = read_wfdb16(header)
        canonical, metadata, phase, scale = reconstruct_document(record, case)
        assert digest(bank/case['file']) == case['sha256']
        with np.load(bank/case['file'], allow_pickle=False) as a:
            image, valid, field = a['image'].copy(), a['valid'].copy(), a['reference_mm'].copy()
        inverse, forward, triangles = reference_field_mesh(field, valid)
        with np.load(comparison/case['query_file']) as a:
            query, reference = a['paper_source'], a['paper_reference']
        predicted = forward(query)
        keep = np.isfinite(predicted).all(1)
        error = np.linalg.norm(predicted[keep]-reference[keep], axis=1)
        write_json(case_out/'reference-mesh-validation.json', {
            'triangles': triangles, 'query_coverage': float(keep.mean()),
            'p95_mm': float(np.quantile(error, .95)) if len(error) else None,
            'max_mm': float(error.max()) if len(error) else None,
            'scale': scale, 'phase_mm': phase.tolist(), 'print_ppm': float(metadata['ppm']),
            'gain_mm_mv': float(metadata['gain']), 'speed_mm_s': float(metadata['speed'])})
        for method in protocol['methods']:
            if method == 'E6-regenerated-canonical':
                rgb, support = canonical, np.ones(canonical.shape[:2])
            else:
                subgrid = 1 if method.endswith('bilinear') else 2
                rgb, support = sample_reference_rgb(image, inverse, metadata, phase, scale, subgrid)
            Image.fromarray(rgb).save(case_out/(method+'.png'))
            probability = trace_probability(model, rgb)
            instances, samples = extract(probability, support, metadata, scale, record)
            expected = sum(r['expected_samples'] for r in instances)
            observed = sum(r['observed_samples'] for r in instances)
            sse = sum(r['mse_mv2']*r['observed_samples'] for r in instances if r['mse_mv2'] is not None)
            row = {'key': key, 'group_id': case['group_id'], 'backend': case['backend'],
                'layout': case['layout'], 'method': method, 'expected_samples': expected,
                'observed_samples': observed, 'coverage': observed/expected if expected else 0.,
                'rmse_mv': float(np.sqrt(sse/observed)) if observed else None,
                'instances': instances}
            all_rows.append(row)
            np.savez_compressed(case_out/(method+'.npz'),
                **{i+'_'+name: value for i, sample in samples.items() for name, value in sample.items()})
            write_json(out/'cases.json', all_rows)
        print(json.dumps({'signal_case': key, 'completed': index+1, 'scheduled': len(selected),
                          'seconds': round(time.perf_counter()-start_time, 2)}), flush=True)
    write_json(out/'complete.json', {'scheduled': len(selected), 'case_method_results': len(all_rows), 'complete': True})
    return all_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bank', 'comparison', 'sources', 'upstream', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--limit', type=int)
    run(**vars(parser.parse_args()))
