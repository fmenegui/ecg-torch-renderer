"""Predeclared sampling-only development probe; no frozen source is modified.

The closed-form reference integrates an ideal strip over a physical pixel box.
Candidates integrate a reconstruction of the supplied finite uint8 source.
These are different objects: observation and quantization error remain visible.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time
import types

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL_SHA256 = '85adf1e4fe3f37b14fff7d6f32aea98e1db2828917e732bd7e3404653e5cd58a'
CELL_SHA256 = '566253fba786de84475f00d593b80876b3869499d3703c18a71c9a5396cf37d6'
ARMS = ['native-bilinear-point'] + [f'area-{k}-{n}' for k in ('bilinear', 'boxhold') for n in (2, 4, 8)]


def cdf_sum_uniforms(threshold, centre_projection, a, b):
    """Exact rectangular-pixel area below a line, via independent uniforms."""
    a, b = abs(float(a)), abs(float(b))
    d = threshold - centre_projection
    if a < 1e-12 and b < 1e-12:
        return (d >= 0).astype(float)
    if a < 1e-12:
        return np.clip((d + b/2)/b, 0, 1)
    if b < 1e-12:
        return np.clip((d + a/2)/a, 0, 1)
    z = d + (a+b)/2
    # Positive-part inclusion/exclusion is the analytical CDF of a+b uniforms.
    value = (np.maximum(z, 0)**2 - np.maximum(z-a, 0)**2
             - np.maximum(z-b, 0)**2 + np.maximum(z-a-b, 0)**2)/(2*a*b)
    return np.clip(value, 0, 1)


def strip_area(centre_projection, a, b, line_centre, width):
    high = cdf_sum_uniforms(line_centre+width/2, centre_projection, a, b)
    low = cdf_sum_uniforms(line_centre-width/2, centre_projection, a, b)
    return np.clip(high-low, 0, 1)


def load_native(geometry_dir, expected_sha):
    sources = {'complete_rectifier.py': expected_sha, 'cell_mesh.py': CELL_SHA256}
    for name, sha in sources.items():
        if hashlib.sha256((geometry_dir/name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Frozen source mismatch: {name}')
    package = types.ModuleType('sampling_frozen_m2')
    package.__path__ = [str(geometry_dir)]
    sys.modules[package.__name__] = package
    module = importlib.import_module('sampling_frozen_m2.complete_rectifier')
    return module.CompleteMap, module.render_complete


def make_mapping(complete_map, source_per_mm):
    a = np.asarray(source_per_mm, float)
    q = np.linalg.inv(a)
    src = np.array([[0., 0.], [200., 0.], [200., 200.], [0., 200.]])
    target = (src-100) @ q.T+10
    canonical = np.array([[12., 12.], [17., 12.], [17., 17.], [12., 17.]])
    observed = (canonical-10) @ a.T+100
    ids = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    mapping = complete_map(src, target, np.empty((0, 2), int),
                           np.array([[0, 1, 2], [0, 2, 3]]),
                           {'method': 'Exact analytical affine sampling control'},
                           observed, canonical, ids, [[0, 0]])
    return mapping, a, q


def output_frame(mapping, density):
    lower = np.floor(mapping.target.min(0)*density)-1
    upper = np.ceil(mapping.target.max(0)*density)+1
    width, height = (upper-lower+1).astype(int)
    yy, xx = np.mgrid[:height, :width]
    physical = (np.stack((xx, yy), -1)+lower)/density
    return lower, physical


def prepare_area_maps(mapping, physical, density, n, shape):
    started = time.perf_counter()
    arrays = []
    offsets = (np.arange(n)+.5)/n-.5
    for oy in offsets:
        for ox in offsets:
            source = mapping.inverse((physical+[ox/density, oy/density]).reshape(-1, 2))
            finite = np.isfinite(source).all(1)
            source[~finite] = -100
            arrays.append(source.reshape(*physical.shape[:2], 2).astype(np.float32))
    arrays = np.asarray(arrays)
    return arrays, {
        'subgrid': n, 'samples_per_output_pixel': n*n,
        'preparation_wall_ms': 1000*(time.perf_counter()-started),
        'map_cache_bytes': arrays.nbytes,
        'inverse_queries': int(n*n*np.prod(physical.shape[:2])),
        'source_shape': list(shape),
    }


def area_sample(rgb, sample_maps, reconstruction):
    """One composed original-source integral, without an intermediate raster.

    Float conversion prevents repeated uint8 quantization. The output is still
    uint8 and is rounded once, after all bounded subpixel contributions.
    """
    source = rgb.astype(np.float32)
    interpolation = cv2.INTER_LINEAR if reconstruction == 'bilinear' else cv2.INTER_NEAREST
    result = np.zeros((*sample_maps.shape[1:3], 3), np.float32)
    for coordinates in sample_maps:
        result += cv2.remap(source, coordinates[..., 0], coordinates[..., 1], interpolation,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return np.clip(np.rint(result/len(sample_maps)), 0, 255).astype(np.uint8)


def score_image(image, roi, reference, density):
    darkness = (255-image[..., 0][roi].astype(float))/255
    error = darkness-reference
    energy = float(darkness.sum()/density**2)
    truth_energy = float(reference.sum()/density**2)
    return {
        'complete_disappearance': bool(darkness.max() == 0),
        'peak_darkness': float(darkness.max()),
        'reference_peak_darkness': float(reference.max()),
        'peak_darkness_absolute_error': float(abs(darkness.max()-reference.max())),
        'pixel_MAE': float(np.abs(error).mean()),
        'pixel_RMSE': float(np.sqrt(np.mean(error**2))),
        'pixel_max_abs_error': float(np.abs(error).max()),
        'integrated_dark_area_mm2': energy,
        'reference_dark_area_mm2': truth_energy,
        'relative_line_energy_error': float((energy-truth_energy)/truth_energy),
        'dark_energy_outside_nonzero_ideal_pixels': float(darkness[reference < 1e-10].sum()/density**2),
    }, np.rint(darkness*255).astype(np.uint8)


def numerical_controls():
    checks = 0
    x = np.array([-2., -.5, 0., .5, 2.])
    np.testing.assert_allclose(cdf_sum_uniforms(0, x, 1, 0), [1, 1, .5, 0, 0]); checks += 1
    # Half a diagonal square, and a quarter-unit right triangle at x+y<=.5.
    np.testing.assert_allclose(cdf_sum_uniforms(0, np.array([0.]), 1, 1), [.5]); checks += 1
    np.testing.assert_allclose(cdf_sum_uniforms(-.5, np.array([0.]), 1, 1), [.125]); checks += 1
    source_columns = np.arange(201)/10
    coverage = strip_area(source_columns, .1, 0, 7.1, .1)
    np.testing.assert_allclose(coverage.sum(), 1, atol=1e-11); checks += 1
    output_columns = np.arange(2., 12.001, .25)
    coverage = strip_area(output_columns, .25, 0, 7.1, .1)
    np.testing.assert_allclose(coverage.sum()*.25, .1, atol=1e-11); checks += 1
    return checks


def contact_sheet(examples, out):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    cell, rowheight, left, top = 156, 172, 220, 66
    canvas = Image.new('RGB', (left+cell*8, top+rowheight*3+42), '#f0f1f3')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), 'Sampling component | same 4 px/mm | fixed physical crop [5.5, 8.5] mm', fill='black', font=font)
    labels = ['Exact area reference']+ARMS
    for col, label in enumerate(labels):
        label = label.replace('area-', 'area\n').replace('-point', '\npoint')
        draw.multiline_text((left+col*cell+3, 28), label, fill='black', font=small, spacing=1)
    for row, example in enumerate(examples):
        y = top+row*rowheight
        label = f"width 0.1 mm\nnormal {example['angle']} deg\nphase {example['offset']} mm"
        draw.multiline_text((12, y+24), label, fill='black', font=font, spacing=5)
        for col, dark in enumerate(example['patches']):
            im = Image.fromarray(np.clip(np.rint(255*(1-dark)), 0, 255).astype(np.uint8)).convert('RGB')
            im = im.resize((130, 130), Image.Resampling.NEAREST)
            canvas.paste(im, (left+col*cell+3, y+3))
            draw.text((left+col*cell+3, y+138), f'peak {np.max(dark):.3f}', fill='black', font=small)
    draw.text((12, canvas.height-28), 'Display enlargement only. Lower peaks can reflect correct pixel-area averaging; no geometry or bank score is shown.', fill='black', font=small)
    canvas.save(out)


def run(out, geometry_dir):
    protocol_path = HERE/'protocol.json'
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('Protocol changed after freeze')
    protocol = json.loads(protocol_path.read_text())
    complete_map, native_render = load_native(geometry_dir, protocol['native_source_sha256'])
    cv2.setNumThreads(1)
    checks = numerical_controls()
    density = protocol['scoring']['density_pixels_per_mm']
    started_all = time.perf_counter()
    records, map_costs, constants, examples = [], {}, [], []
    sy, sx = np.mgrid[:201, :201]
    source_xy = np.stack((sx, sy), -1).astype(float)
    out.mkdir(parents=True, exist_ok=True)
    for map_spec in protocol['maps']:
        mapping, a, q = make_mapping(complete_map, map_spec['source_pixels_per_physical_mm_matrix'])
        lower, physical = output_frame(mapping, density)
        flat = physical.reshape(-1, 2)
        # Interior score centres are independent of every reconstructed image.
        roi = ((physical >= 2-1e-10) & (physical <= 12+1e-10)).all(-1)
        roi_xy = physical[roi]
        assert roi.sum() == 41*41
        np.testing.assert_allclose(mapping.inverse(roi_xy), (roi_xy-10) @ a.T+100, atol=2e-12); checks += 1
        maps, costs = {}, []
        for n in (2, 4, 8):
            maps[n], cost = prepare_area_maps(mapping, physical, density, n, (201, 201, 3))
            # Every subpixel footprint scored here is inside the source frame.
            assert (maps[n][:, roi] > 0).all() and (maps[n][:, roi] < 200).all(); checks += 1
            costs.append(cost)
        map_costs[map_spec['id']] = costs
        rgb_constant = np.empty((201, 201, 3), np.uint8)
        rgb_constant[:] = protocol['constant_control_rgb']
        native, valid, _, metadata = native_render(rgb_constant, mapping, pixels_per_mm=density)
        np.testing.assert_allclose(metadata['output_origin_mm'], lower/density); checks += 1
        assert native.shape[:2] == physical.shape[:2] and valid[roi].all(); checks += 1
        for arm in ARMS:
            if arm == ARMS[0]:
                image = native
            else:
                _, kind, n = arm.split('-')
                image = area_sample(rgb_constant, maps[int(n)], kind)
            err = int(np.abs(image[roi].astype(int)-rgb_constant[0, 0].astype(int)).max())
            assert err == 0; checks += 1
            constants.append({'map': map_spec['id'], 'arm': arm, 'max_channel_error': err})
        source_physical = (source_xy-100) @ q.T+10
        predictions, references, source_grays, scene_meta = [], [], [], []
        for width in protocol['thin_strip_scenes']['strip_width_mm']:
            for angle in protocol['thin_strip_scenes']['normal_angles_degrees']:
                n = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                source_normal = q.T @ n
                for offset in protocol['thin_strip_scenes']['offsets_mm']:
                    line_centre = n @ np.array([7., 7.])+offset
                    observation = strip_area(source_physical @ n, abs(source_normal[0]),
                                             abs(source_normal[1]), line_centre, width)
                    gray = np.clip(np.rint(255*(1-observation)), 0, 255).astype(np.uint8)
                    rgb = np.repeat(gray[..., None], 3, -1)
                    reference_full = strip_area(physical @ n, abs(n[0])/density,
                                                abs(n[1])/density, line_centre, width)
                    reference = reference_full[roi]
                    scene_predictions = []
                    example = (map_spec['id'] == 'affine-isotropic' and width == .1
                               and ((angle == 0 and offset in (.1, .4)) or (angle == 45 and offset == .15)))
                    crop = ((physical >= 5.5-1e-10) & (physical <= 8.5+1e-10)).all(-1)
                    patches = [reference_full[crop].reshape(13, 13)] if example else []
                    for arm in ARMS:
                        started = time.perf_counter()
                        if arm == ARMS[0]:
                            image, _, _, meta = native_render(rgb, mapping, pixels_per_mm=density)
                            assert meta['resampled']
                        else:
                            _, kind, size = arm.split('-')
                            image = area_sample(rgb, maps[int(size)], kind)
                        elapsed = 1000*(time.perf_counter()-started)
                        metrics, dark_u8 = score_image(image, roi, reference, density)
                        records.append({'map': map_spec['id'], 'width_mm': width,
                                        'normal_angle_degrees': angle, 'offset_mm': offset,
                                        'arm': arm, 'application_wall_ms': elapsed, **metrics})
                        scene_predictions.append(dark_u8.reshape(41, 41))
                        if example:
                            patches.append((255-image[..., 0][crop].astype(float)).reshape(13, 13)/255)
                    if example:
                        examples.append({'angle': angle, 'offset': offset, 'patches': patches})
                    predictions.append(scene_predictions)
                    references.append(reference.reshape(41, 41))
                    source_grays.append(gray)
                    scene_meta.append([width, angle, offset])
        np.savez_compressed(out/f"{map_spec['id']}.npz", arms=np.array(ARMS),
                            scene_width_angle_offset=np.array(scene_meta), roi_xy_mm=roi_xy.reshape(41, 41, 2),
                            darkness_u8=np.asarray(predictions), ideal_coverage_f64=np.asarray(references),
                            source_gray_u8=np.asarray(source_grays), source_per_mm=a)
    summaries = []
    for width in (.1, 1.):
        for arm in ARMS:
            subset = [r for r in records if r['width_mm'] == width and r['arm'] == arm]
            summaries.append({'width_mm': width, 'arm': arm, 'cases': len(subset),
                'disappearances': sum(r['complete_disappearance'] for r in subset),
                'mean_pixel_MAE': float(np.mean([r['pixel_MAE'] for r in subset])),
                'worst_pixel_MAE': max(r['pixel_MAE'] for r in subset),
                'mean_abs_relative_energy_error': float(np.mean([abs(r['relative_line_energy_error']) for r in subset])),
                'worst_abs_relative_energy_error': max(abs(r['relative_line_energy_error']) for r in subset),
                'mean_peak_abs_error': float(np.mean([r['peak_darkness_absolute_error'] for r in subset])),
                'minimum_peak_darkness': min(r['peak_darkness'] for r in subset),
                'median_application_wall_ms': float(np.median([r['application_wall_ms'] for r in subset]))})
    report = {'schema': 'm3-sampling-component-results-v1', 'protocol_sha256': PROTOCOL_SHA256,
              'candidate_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'native_source_sha256': protocol['native_source_sha256'],
              'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
              'numerical_checks': checks, 'scene_count': len(records)//len(ARMS),
              'arms': ARMS, 'results_count': len(records), 'summaries': summaries,
              'constant_controls': constants, 'map_preparation_cost': map_costs, 'cases': records,
              'total_wall_seconds': time.perf_counter()-started_all,
              'scope': protocol['limits']}
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    contact_sheet(examples, out/'contact-sheet.png')
    print(json.dumps({'out': str(out), 'scenes': report['scene_count'], 'results': len(records),
                      'checks': checks, 'wall_seconds': report['total_wall_seconds'],
                      'summaries': summaries}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'results')
    parser.add_argument('--geometry-dir', type=Path, default=ROOT/'ecg-torch-renderer/ecg_gpu/digitization')
    args = parser.parse_args()
    run(args.out, args.geometry_dir)
