"""RGB-only inference with a diverse-study multitask checkpoint.

The optional restored image is a diagnostic. The default output preserves the
full input frame, with inferred geometry and a separate evidence map. Explicit
cell cutouts are available only as a diagnostic mode.
"""
import argparse
import json
from numbers import Real
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.artifacts import sha256_file
from ecg_gpu.digitization.denoising_grid import DenoisingGridNet
from ecg_gpu.digitization.complete_rectifier import fit_complete_map, render_complete
from ecg_gpu.digitization.cell_mesh import CellMap, cell_triangles
from ecg_gpu.digitization.lattice import organize_lattice
from ecg_gpu.digitization.cycle_lattice import organize_cycle_lattice
from ecg_gpu.digitization.observation import remove_periodic_banding
from ecg_gpu.digitization.local_cells import locally_supported_cells
from ecg_gpu.digitization.support import cell_support_summary
from ecg_gpu.digitization.runner import _write_json
from ecg_gpu.digitization.robust_refinement import refine_grid_points
from ecg_gpu.grid_point_learning import decode_grid_points
from examples.digitization_grid_rectification import grid_remap


def _refinement_strength(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('Refinement strength must be finite and in [0,1]')
    return float(value)


def _fit_refined_geometry(observed, ids, scores, cells, shape, strength, out):
    """Try the opt-in candidate, then the unchanged baseline on input failure.

    No gold or signal reference enters this decision. A candidate is accepted
    only through the existing full-frame geometric validation. That validation
    is an implementation contract, not proof of improved physical accuracy.
    """
    report = {'requested_strength': strength, 'crease_guard_enabled': True,
              'applied': False, 'gold_used': False, 'fallback_to_baseline': False,
              'fallback_count': 0, 'fallback_reason': None,
              'effective_positions': 'original detector positions',
              'original_detection_positions_preserved': True,
              'original_detection_data': 'detected-grid-geometry.npz',
              'candidate_data': None, 'candidate_fit': None}
    try:
        candidate, diagnostics = refine_grid_points(observed.copy(), ids.copy(), scores.copy(),
            strength=strength, crease_guard=True)
        report['refiner'] = diagnostics
        candidate = np.asarray(candidate)
        np.savez_compressed(out/'refined-grid-candidate.npz', xy=candidate, scores=scores,
                            ids=ids, canonical_mm=ids[:, ::-1]*5., cells=np.asarray(cells, int).reshape(-1, 2))
        report['candidate_data'] = 'refined-grid-candidate.npz'
        changed = not np.array_equal(candidate, observed)
        report['candidate_positions_changed'] = changed
        if changed:
            try:
                mapping, fitted = fit_complete_map(candidate, ids[:, ::-1]*5., ids,
                                                   shape, scores, cells)
            except Exception as error:
                # Optional geometry backends have their own exception types.
                # Keep this candidate-only catch outside the baseline call.
                mapping = None
                fitted = {'status': 'error', 'reason': type(error).__name__+': '+str(error)}
            report['candidate_fit'] = fitted
            if mapping is not None:
                report.update(applied=True, effective_positions='refined detector positions',
                              original_detection_positions_preserved=False)
                return mapping, fitted, candidate, report
            report.update(fallback_to_baseline=True, fallback_count=1,
                          fallback_reason='refined full-frame fit rejected: '+str(fitted.get('reason')))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        report.update(fallback_to_baseline=True, fallback_count=1,
                      fallback_reason=type(error).__name__+': '+str(error))
        report['candidate_fit'] = {'status': 'error', 'reason': report['fallback_reason']}
    mapping, fitted = fit_complete_map(observed, ids[:, ::-1]*5., ids, shape, scores, cells)
    return mapping, fitted, observed, report


def write_geometry_output(rgb, xy, scores, local, out, *, pixels_per_mm=8., output_mode='full',
                          refinement_strength=0.):
    """Preserve original visible content independently of detected-cell evidence."""
    if output_mode not in ('full', 'cells'):
        raise ValueError('Output mode must be full or cells')
    refinement_strength = _refinement_strength(refinement_strength)
    if refinement_strength > 0 and output_mode == 'cells':
        raise ValueError('Soft refinement requires full output mode')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = np.asarray(local['selected_indices'], int)
    ids = np.asarray(local['ids'], int).reshape(-1, 2)
    observed = xy[chosen]
    report = {'output_mode': output_mode, 'source_for_rectification': 'original RGB'}
    Image.fromarray(rgb).save(out/'original-rgb.png')
    if refinement_strength > 0:
        np.savez_compressed(out/'detected-grid-geometry.npz', xy=observed, scores=scores[chosen],
                            ids=ids, canonical_mm=ids[:, ::-1]*5.,
                            cells=np.asarray(local['cells'], int).reshape(-1, 2), selected_indices=chosen)
        mapping, fitted, observed, refinement = _fit_refined_geometry(observed, ids, scores[chosen],
            local['cells'], rgb.shape, refinement_strength, out)
        report['point_refinement'] = refinement
    np.savez_compressed(out/'observed-grid-geometry.npz', xy=observed, scores=scores[chosen],
                        ids=ids, canonical_mm=ids[:, ::-1]*5.,
                        cells=np.asarray(local['cells'], int).reshape(-1, 2))
    if output_mode == 'cells':
        report['status'] = 'rejected'
        if local['partial_status'] == 'partial':
            canonical = ids[:, ::-1]*(5*pixels_per_mm)
            size = (canonical.max(0)+1).astype(int)[::-1]
            if max(size) <= 4096:
                warped, support = grid_remap(rgb, xy[chosen], canonical, ids,
                                            tuple(size), allowed_cells=local['cells'])
                Image.fromarray(warped).save(out/'rectified-original-rgb.png')
                Image.fromarray(support.astype(np.uint8)*255).save(out/'output-support.png')
                report.update(status='partial-rectification',
                              supported_output_canvas_fraction=float(support.mean()))
        return report
    if refinement_strength == 0:
        mapping, fitted = fit_complete_map(observed, ids[:, ::-1]*5., ids, rgb.shape,
                                          scores[chosen], local['cells'])
    warped, valid, evidence, rendered, eligible = render_complete(rgb, mapping,
        pixels_per_mm=pixels_per_mm, return_eligibility=True)
    details = dict(fitted, **rendered)
    if refinement_strength > 0:
        changed = report['point_refinement']['applied']
        details.update(effective_input_cell_map_changed=fitted.get('observed_cell_map_changed'),
                       observed_cell_map_changed=changed,
                       observed_cell_map_changed_reference='original detector positions',
                       original_detection_positions_preserved=not changed,
                       effective_observation_positions=report['point_refinement']['effective_positions'])
        if changed:
            details['method'] = 'constrained completion preserving supplied refined-cell triangles'
    if not details['resampled'] and len(ids):
        triangles = cell_triangles(observed, ids[:, ::-1]*5., ids, local['cells'])
        partial = CellMap(observed, ids[:, ::-1]*5., triangles)
        for start in range(0, rgb.shape[0], 48):
            yy, xx = np.mgrid[start:min(start+48, rgb.shape[0]), :rgb.shape[1]]
            keep = np.isfinite(partial(np.column_stack((xx.ravel(), yy.ravel())))).all(1)
            eligible[start:start+len(yy)].reshape(-1)[keep] = True
            evidence[start:start+len(yy)].reshape(-1)[keep] = 3
        details['observed_cell_coordinates_available'] = bool(eligible.any())
    filename = 'rectified-original-rgb.png' if details['resampled'] else 'original-preserved.png'
    Image.fromarray(warped).save(out/filename)
    Image.fromarray(warped).save(out/'output-original-rgb.png')
    Image.fromarray(valid.astype(np.uint8)*255).save(out/'sampling-valid.png')
    colors = np.array([[225, 229, 231], [227, 150, 85], [104, 160, 207], [49, 145, 109]], np.uint8)
    Image.fromarray(colors[evidence]).save(out/'geometry-evidence.png')
    Image.fromarray(eligible.astype(np.uint8)*255).save(out/'geometry-eligibility.png')
    np.savez_compressed(out/'geometry-evidence.npz', sampling_valid=valid, evidence=evidence,
                        geometric_eligibility=eligible)
    if mapping is not None:
        np.savez_compressed(out/'full-frame-mesh.npz', source=mapping.source,
                            target_mm=mapping.target, triangles=mapping.triangles)
    report.update(status='rectification-proposal' if details['resampled'] else 'original-preserved',
        output_image=filename, complete_geometry=details, original_image='original-rgb.png',
        geometry_accuracy_verified=False, evidence_map_erases_rgb=False,
        sampled_map_roundtrip_checked=False,
        coordinate_mode='complete-map-with-eligibility' if details['resampled'] else 'observed-cells-only',
        observed_cell_coordinate_data='observed-grid-geometry.npz',
        evidence_legend={'0': 'outside sampled frame or original unrectified',
                         '1': 'extrapolated geometry', '2': 'interpolated geometry',
                         '3': 'observed grid cell, not an accuracy certificate'})
    if refinement_strength > 0 and report['point_refinement']['applied']:
        report['evidence_legend']['3'] = 'cell fitted from refined detector positions, not an accuracy certificate'
    return report


def rectify(checkpoint, image, out, device='cpu', pixels_per_mm=8., profile='banding', output_mode='full',
            refinement_strength=0.):
    refinement_strength = _refinement_strength(refinement_strength)
    if refinement_strength > 0 and output_mode == 'cells':
        raise ValueError('Soft refinement requires full output mode')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory')
    if not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError('Positive finite output density required')
    if output_mode not in ('full', 'cells'):
        raise ValueError('Output mode must be full or cells')
    if profile not in ('baseline', 'threshold-70', 'banding', 'banding-recall', 'geometry-recall'):
        raise ValueError('Unknown geometric inference profile')
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
    banding = {'applied': False}
    point_threshold = float(state['threshold'])
    fallback = None
    if profile != 'baseline':
        pilot_xy, pilot_scores = xy.copy(), scores.copy()
        filtered = rgb
        if profile != 'threshold-70':
            filtered, banding = remove_periodic_banding(rgb, pilot_xy)
        geometry_output = output
        if banding['applied']:
            canvas[:rh, :rw] = np.asarray(Image.fromarray(filtered).resize((rw, rh), Image.Resampling.BILINEAR))
            tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1)[None].float().to(device)/255
            with torch.inference_mode():
                geometry_output = model(tensor)
        point_threshold = .5 if profile == 'geometry-recall' else .7 if profile in ('threshold-70', 'banding-recall') else float(state['threshold'])
        prediction = decode_grid_points(geometry_output['heat'].float().sigmoid(), geometry_output['offsets'].float(),
                                        threshold=point_threshold, max_points=16384)
        keep = prediction['valid'][0]
        xy = prediction['xy'][0, keep].cpu().numpy()
        scores = prediction['scores'][0, keep].cpu().numpy()
        inside = ((xy >= 0) & (xy <= [rw-1, rh-1])).all(1)
        xy, scores = (xy[inside]+.5)/[rw/w, rh/h]-.5, scores[inside]
        high = scores >= float(state['threshold'])
        hint = organize_lattice(xy[high], scores[high])
        if profile != 'geometry-recall':
            native = organize_lattice(xy, scores)
        elif hint.get('spacing_px'):
            native = organize_cycle_lattice(xy, scores, geometry_hint=hint)
        else:
            xy, scores = pilot_xy, pilot_scores
            native = organize_lattice(xy, scores)
            fallback = 'original pilot, unavailable geometry hint'
    else:
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
    report.update(detection_profile=profile, point_threshold=point_threshold,
                  background_banding=banding, inference_fallback=fallback,
                  sampled_map_roundtrip_checked=True)
    report['cell_support'] = cell_support_summary(local)
    np.savez_compressed(out/'points.npz', xy=xy, scores=scores,
                        selected=np.asarray(local['selected_indices']), ids=np.asarray(local['ids']), cells=np.asarray(local['cells']))
    if state['joint']:
        restored = Image.fromarray((clean*255).round().clip(0, 255).astype(np.uint8))
        restored.resize((w, h), Image.Resampling.BILINEAR).save(out/'restored-diagnostic.png')
    report.update(write_geometry_output(rgb, xy, scores, local, out,
        pixels_per_mm=pixels_per_mm, output_mode=output_mode, refinement_strength=refinement_strength))
    _write_json(out/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'image', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--pixels-per-mm', type=float, default=8.)
    parser.add_argument('--profile', choices=('baseline', 'threshold-70', 'banding', 'banding-recall', 'geometry-recall'), default='banding')
    parser.add_argument('--output-mode', choices=('full', 'cells'), default='full',
                        help='Full original-RGB frame with separate evidence, or cell-only diagnostic')
    parser.add_argument('--refinement-strength', type=float, default=0.,
                        help='Opt-in soft point refinement in [0,1], default 0 preserves original detections')
    torch.set_num_threads(4)
    print(json.dumps(rectify(**vars(parser.parse_args()))))
