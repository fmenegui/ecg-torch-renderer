"""Opt-in global and native-pixel grid observation, with bounded refinement.

Only RGB and the model's global detections determine the ROI. Local tiles are
cropped, never resized. Their heatmaps and offsets are fused before decoding.
This is an inference experiment, not equivalence to full-frame GroupNorm.
The caller must keep the original RGB for final geometric sampling.
"""
import math
from numbers import Real
import time

import numpy as np
from PIL import Image
import torch

from ..grid_point_learning import decode_grid_points


def _integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be an integer in the supported range')


def _real(value, name, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f'{name} must be finite and in the supported range')


def _positions(length, side, overlap):
    """Cover an axis and include its last pixel, with an explicit final tile."""
    if length <= side:
        return [0]
    result = list(range(0, length-side+1, side-overlap))
    if result[-1] != length-side:
        result.append(length-side)
    return result


def _probabilities(model, rgb, device):
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
    output = model(tensor.to(device=device, dtype=torch.float32)/255)
    if not isinstance(output, dict) or 'heat' not in output or 'offsets' not in output:
        raise ValueError('Model must return heat logits and same-resolution offsets')
    heat, offsets = output['heat'], output['offsets']
    height, width = rgb.shape[:2]
    if (not isinstance(heat, torch.Tensor) or not isinstance(offsets, torch.Tensor)
            or heat.shape != (1, 1, height, width) or offsets.shape != (1, 2, height, width)):
        raise ValueError('Model heads must preserve input pixel resolution and xy offset order')
    if not torch.isfinite(heat).all() or not torch.isfinite(offsets).all():
        raise ValueError('Finite model logits and offsets required')
    # Conversion to CPU is an explicit inference boundary, outside training.
    return heat.float().sigmoid().cpu(), offsets.float().cpu()


def _decode(heat, offsets, threshold, max_points, nms_radius):
    prediction = decode_grid_points(heat, offsets, threshold=threshold,
                                    max_points=max_points, nms_radius=nms_radius)
    keep = prediction['valid'][0]
    return (prediction['xy'][0, keep].numpy(), prediction['scores'][0, keep].numpy(),
            bool(prediction['truncated'][0]))


def _deduplicate(xy, scores, origins, radius, capacity):
    """Greedy score ranking with bounded per-query neighbor materialization."""
    from scipy.spatial import cKDTree

    if not len(xy):
        return xy.astype(np.float32), scores.astype(np.float32), origins, 0, False
    tree = cKDTree(xy)
    suppressed = np.zeros(len(xy), dtype=bool)
    order = np.argsort(-scores, kind='stable')
    chosen = []
    for index in order:
        if not suppressed[index]:
            chosen.append(index)
            suppressed[tree.query_ball_point(xy[index], radius)] = True
    count = len(chosen)
    chosen = np.asarray(chosen[:capacity], dtype=int)
    return (xy[chosen].astype(np.float32), scores[chosen].astype(np.float32), origins[chosen],
            count, count > capacity)


@torch.inference_mode()
def detect_native_grid(model, rgb, *, device='cpu', threshold=.85, max_points=16384,
                       nms_radius=2, global_size=(704, 1024), tile_size=512, overlap=128,
                       roi_margin=64, min_roi_points=4, max_tiles=8,
                       max_roi_pixels=4194304, dedup_radius=1.5):
    """Return ``(xy, scores, report)`` in original sensor pixel centers.

    ``global_size`` is (height, width). The global view uses top-left letterbox
    and the half-pixel resize convention. Refinement tiles have native pixels
    with no resizing. A ROI that exceeds either budget falls back to the global
    output as a whole, rather than silently refining only its first tiles.

    Global and native scores are uncalibrated rankings. Greedy deduplication
    keeps the strongest candidate within ``dedup_radius`` original pixels.
    A lower score or different model context can change the selected location.
    This function does not certify pitch, IDs, observability or physical error.
    It does not return restored RGB and does not modify the input image.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB must be a uint8 [H,W,3] array')
    if min(rgb.shape[:2]) < 1:
        raise ValueError('Nonempty RGB required')
    if not isinstance(global_size, (tuple, list)) or len(global_size) != 2:
        raise ValueError('global_size must contain height and width')
    for value in global_size:
        _integer(value, 'global_size', 8, 4096)
    if global_size[0]*global_size[1] > 4194304:
        raise ValueError('Global input exceeds the supported pixel budget')
    _integer(max_points, 'max_points', 1, 32768)
    _integer(nms_radius, 'nms_radius', 1, 16)
    _integer(tile_size, 'tile_size', 16, 2048)
    _integer(overlap, 'overlap', 0, tile_size-1)
    _integer(roi_margin, 'roi_margin', 0, 8192)
    _integer(min_roi_points, 'min_roi_points', 1, 32768)
    _integer(max_tiles, 'max_tiles', 0, 64)
    _integer(max_roi_pixels, 'max_roi_pixels', 1, 16777216)
    _real(threshold, 'threshold', 1e-9, 1.)
    _real(dedup_radius, 'dedup_radius', 0., 16.)

    started = time.perf_counter()
    height, width = rgb.shape[:2]
    gh, gw = global_size
    scale = min(gw/width, gh/height)
    rw, rh = max(1, round(width*scale)), max(1, round(height*scale))
    canvas = np.full((gh, gw, 3), 255, dtype=np.uint8)
    canvas[:rh, :rw] = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BILINEAR))
    prior_training = getattr(model, 'training', None)
    if hasattr(model, 'eval'):
        model.eval()
    try:
        heat, offsets = _probabilities(model, canvas, device)
        # Padding is a known resize artifact, not image evidence. Exclude it
        # before NMS/top-k so its predictions cannot exhaust point capacity.
        global_xy, global_scores, global_truncated = _decode(heat[:, :, :rh, :rw],
            offsets[:, :, :rh, :rw], threshold, max_points, nms_radius)
        global_xy = (global_xy+.5)/[rw/width, rh/height]-.5
        inside = ((global_xy >= 0) & (global_xy <= [width-1, height-1])).all(1)
        global_xy, global_scores = global_xy[inside], global_scores[inside]
        global_seconds = time.perf_counter()-started
        report = {
            'method': 'opt-in global plus native ROI tile inference',
            'input_shape': [height, width], 'global_canvas_shape': [gh, gw],
            'global_resized_shape': [rh, rw], 'global_scale_xy': [rw/width, rh/height],
            'pixel_convention': 'source=(resized+0.5)/scale-0.5, tiles use integer translation',
            'global_points': len(global_xy), 'native_points': 0,
            'global_decoder_truncated': global_truncated, 'native_decoder_truncated': False,
            'dedup_capacity_truncated': False, 'roi_xyxy_exclusive': None,
            'roi_source': 'global model points only', 'native_tiles': [],
            'required_native_tiles': 0, 'native_input_pixels': 0,
            'max_tiles': max_tiles, 'max_roi_pixels': max_roi_pixels,
            'tile_size': tile_size, 'overlap': overlap, 'dedup_radius_source_px': float(dedup_radius),
            'native_resized': False, 'budget_exhausted': False, 'fallback_reason': None,
            'full_tile_equivalence_claimed': False, 'confidence_calibrated': False,
            'gold_used_for_inference': False, 'original_rgb_modified': False,
            'global_seconds': global_seconds,
        }
        native_xy = np.empty((0, 2), dtype=np.float32)
        native_scores = np.empty(0, dtype=np.float32)
        if max_tiles == 0:
            report['fallback_reason'] = 'native refinement disabled'
        elif len(global_xy) < min_roi_points:
            report['fallback_reason'] = 'too few global points to propose a ROI'
        else:
            low = np.floor(global_xy.min(0)).astype(int)-roi_margin
            high = np.ceil(global_xy.max(0)).astype(int)+roi_margin+1
            x0, y0 = np.maximum(low, [0, 0])
            x1, y1 = np.minimum(high, [width, height])
            x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
            report['roi_xyxy_exclusive'] = [x0, y0, x1, y1]
            roi_h, roi_w = y1-y0, x1-x0
            xs, ys = _positions(roi_w, tile_size, overlap), _positions(roi_h, tile_size, overlap)
            required = len(xs)*len(ys)
            report['required_native_tiles'] = required
            if min(roi_h, roi_w) < 8:
                report['fallback_reason'] = 'native ROI is smaller than model support'
            elif required > max_tiles or roi_h*roi_w > max_roi_pixels:
                report.update(budget_exhausted=True, fallback_reason='complete native ROI exceeds budget')
            else:
                heat_sum = torch.zeros((1, 1, roi_h, roi_w), dtype=torch.float32)
                offset_sum = torch.zeros((1, 2, roi_h, roi_w), dtype=torch.float32)
                weight_sum = torch.zeros((1, 1, roi_h, roi_w), dtype=torch.float32)
                for top in ys:
                    for left in xs:
                        bottom, right = min(top+tile_size, roi_h), min(left+tile_size, roi_w)
                        patch = rgb[y0+top:y0+bottom, x0+left:x0+right]
                        local_heat, local_offsets = _probabilities(model, patch, device)
                        ph, pw = patch.shape[:2]
                        # Positive feathering preserves outer ROI pixels. It
                        # downweights patch edges without creating blank seams.
                        wy = torch.sin(math.pi*(torch.arange(ph)+.5)/ph).clamp_min(.05)
                        wx = torch.sin(math.pi*(torch.arange(pw)+.5)/pw).clamp_min(.05)
                        weights = (wy[:, None]*wx[None])[None, None]
                        heat_sum[:, :, top:bottom, left:right] += local_heat*weights
                        offset_sum[:, :, top:bottom, left:right] += local_offsets*weights
                        weight_sum[:, :, top:bottom, left:right] += weights
                        report['native_tiles'].append([x0+left, y0+top, x0+right, y0+bottom])
                        report['native_input_pixels'] += ph*pw
                if not (weight_sum > 0).all():
                    raise RuntimeError('Native ROI coverage has an internal hole')
                native_xy, native_scores, truncated = _decode(heat_sum/weight_sum,
                    offset_sum/weight_sum, threshold, max_points, nms_radius)
                native_xy += [x0, y0]
                report['native_points'] = len(native_xy)
                report['native_decoder_truncated'] = truncated
        xy = np.vstack((global_xy, native_xy))
        scores = np.concatenate((global_scores, native_scores))
        origins = np.concatenate((np.zeros(len(global_xy), dtype=np.uint8),
                                  np.ones(len(native_xy), dtype=np.uint8)))
        if len(native_xy):
            xy, scores, origins, count, truncated = _deduplicate(xy, scores, origins, dedup_radius, max_points)
        else:
            # Without any new observation, preserve the global decoder output.
            # Source-space dedup could otherwise merge two globally decoded
            # peaks after strong upscaling of a very small source image.
            xy, scores = xy.astype(np.float32), scores.astype(np.float32)
            count, truncated = len(xy), False
        report.update(unique_candidates_before_capacity=count, dedup_capacity_truncated=truncated,
                      returned_points=len(xy), returned_native_points=int(origins.sum()),
                      truncated=bool(global_truncated or report['native_decoder_truncated'] or truncated),
                      model_passes=1+len(report['native_tiles']),
                      seconds=time.perf_counter()-started)
        return xy, scores, report
    finally:
        if prior_training is not None and hasattr(model, 'train'):
            model.train(prior_training)
