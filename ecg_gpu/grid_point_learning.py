"""Loss and local subpixel decoding for a grid-point U-Net head.

One channel predicts a center heatmap. Two optional channels predict x/y
offsets at unique nearest-pixel anchors. No global soft-argmax is used: a
page contains many independent points. Output coordinates use pixel centers.
"""
import math
from collections.abc import Mapping
from numbers import Real

import torch
from torch.nn import functional as F


def grid_point_loss(logits, offsets, batch, *, offset_weight=1., pos_weight=None):
    """BCE center loss plus masked Smooth-L1 offsets, at output stride one.

    Pass ``offsets=None`` for a one-channel heatmap network. Otherwise request
    GridPointConfig(return_offsets=True). Empty anchor support contributes zero
    offset loss/gradients. Background negatives outside paper are included, so
    the network must also learn not to detect points on the table.
    """
    if not isinstance(batch, Mapping):
        raise TypeError('batch must be a mapping of grid-point targets')
    if 'grid_points_mask' not in batch:
        raise ValueError('Missing grid_points_mask. Request grid-point supervision first')
    target = batch['grid_points_mask']
    if (not isinstance(logits, torch.Tensor) or not isinstance(target, torch.Tensor)
            or logits.ndim != 4 or min(logits.shape) < 1 or logits.shape[1] != 1
            or logits.shape != target.shape or logits.device != target.device):
        raise ValueError('logits/target must be matching [B,1,H,W] tensors on one device')
    if not logits.is_floating_point() or target.is_complex():
        raise ValueError('logits must be floating point and targets must be real')
    if isinstance(offset_weight, bool) or not isinstance(offset_weight, Real) or not math.isfinite(offset_weight) or offset_weight < 0:
        raise ValueError('offset_weight must be nonnegative and finite')
    if pos_weight is not None:
        if isinstance(pos_weight, bool) or not isinstance(pos_weight, Real) or not math.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError('pos_weight must be positive and finite')
        pos_weight = logits.new_tensor(pos_weight, dtype=torch.float32)
    torch._assert_async((torch.isfinite(target) & (target >= 0) & (target <= 1)).all(), 'grid_points_mask must contain finite values in [0,1]')
    center_loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), pos_weight=pos_weight)
    # Summing float16 logits can overflow even when BCE itself is finite.
    offset_loss = center_loss*0
    if offsets is not None:
        if {'grid_points_offsets', 'grid_points_offset_valid'} - batch.keys():
            raise ValueError('Request return_offsets=True for offset supervision')
        truth, support = batch['grid_points_offsets'], batch['grid_points_offset_valid']
        expected = (logits.shape[0], 2, *logits.shape[-2:])
        if (any(not isinstance(value, torch.Tensor) for value in (offsets, truth, support))
                or offsets.shape != expected or truth.shape != expected or support.shape != logits.shape):
            raise ValueError('Offsets must be [B,2,H,W] and support [B,1,H,W]')
        if any(value.device != logits.device for value in (offsets, truth, support)):
            raise ValueError('All offset tensors must share the logits device')
        if not offsets.is_floating_point() or not truth.is_floating_point() or support.is_complex():
            raise ValueError('Offset predictions/targets must be floating point and support must be real')
        torch._assert_async((torch.isfinite(support) & (support >= 0) & (support <= 1)).all(), 'Offset support must contain finite values in [0,1]')
        support = support.float()
        active = support > 0
        truth = torch.where(active, truth.float(), 0.)
        torch._assert_async((torch.isfinite(truth) & (truth.abs() <= .5+1e-5)).all(), 'Supervised offsets must be finite and in [-0.5,0.5] pixels')
        prediction = torch.where(active, offsets.float(), 0.)
        error = F.smooth_l1_loss(prediction, truth, reduction='none')
        offset_loss = (error*support).sum()/(2*support.sum()).clamp_min(1)
    return {'loss':center_loss+offset_weight*offset_loss, 'heatmap_loss':center_loss, 'offset_loss':offset_loss}


@torch.no_grad()
def decode_grid_points(heatmap, offsets=None, *, threshold=.3, max_points=4096, nms_radius=2):
    """Decode many local peaks to fixed-shape subpixel xy/scores/valid arrays.

    Input is a probability heatmap, not logits. Optional offsets are in pixels
    at the SAME resolution. Without offsets, use a three-sample log-quadratic
    fit per axis, exact for an isolated ideal Gaussian away from boundaries.
    A binary disk/flat plateau does not encode a unique subpixel center.
    Thresholds, NMS radius and capacities need validation for the deployment
    resolution. ``truncated`` flags overflow instead of silently hiding it.
    """
    if heatmap.ndim != 4 or min(heatmap.shape) < 1 or heatmap.shape[1] != 1:
        raise ValueError('heatmap must be [B,1,H,W]')
    if not heatmap.is_floating_point():
        raise ValueError('heatmap must be floating point')
    if type(max_points) is not int or max_points < 1 or type(nms_radius) is not int or not 1 <= nms_radius <= 16:
        raise ValueError('Positive max_points and integer nms_radius in 1..16 required')
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('threshold must be in (0,1]')
    b, _, h, w = heatmap.shape
    if offsets is not None and (offsets.shape != (b, 2, h, w) or offsets.device != heatmap.device):
        raise ValueError('offsets must be [B,2,H,W] on the heatmap device')
    score = heatmap.float()
    torch._assert_async((torch.isfinite(score) & (score >= 0) & (score <= 1)).all(), 'Finite probability heatmap required')
    kernel = 2*nms_radius+1
    peak = (score == F.max_pool2d(score, kernel, 1, nms_radius)) & (score >= threshold)
    # Match torch.round's even-pixel anchors at exact half-pixel centers.
    # float64 keeps integer IDs exact even for images larger than 2^24 pixels.
    ids = torch.arange(h*w, device=score.device, dtype=torch.float64).reshape(1, 1, h, w)
    parity = ids.remainder(w).remainder(2)+ids.div(w, rounding_mode='floor').remainder(2)
    priority = parity*(h*w)+ids
    tie = torch.where(peak, priority, float(3*h*w))
    first = -F.max_pool2d(-tie, kernel, 1, nms_radius)
    peak = peak & (priority == first)
    k = min(max_points, h*w)
    confidence, indices = torch.topk(torch.where(peak, score, -1.).flatten(1), k, dim=1, sorted=True)
    px, py = indices.remainder(w), indices.div(w, rounding_mode='floor')
    xy = torch.stack((px, py), -1).float()
    present = confidence >= threshold
    if offsets is not None:
        delta = offsets.float().flatten(2).gather(2, indices[:, None].expand(-1, 2, -1)).transpose(1, 2)
        present = present & torch.isfinite(delta).all(-1) & (delta.abs().amax(-1) <= .5+1e-5)
    else:
        log_score = score.clamp_min(1e-12).log().flatten(1)
        center = log_score.gather(1, indices)
        steps = []
        for step, inside in ((1, (px > 0) & (px < w-1)), (w, (py > 0) & (py < h-1))):
            left = log_score.gather(1, (indices-step).clamp(0, h*w-1))
            right = log_score.gather(1, (indices+step).clamp(0, h*w-1))
            denominator = 2*(left-2*center+right)
            fit = (left-right)/torch.where(denominator.abs() > 1e-6, denominator, torch.ones_like(denominator))
            steps.append(torch.where(inside & (denominator < -1e-6), fit.clamp(-.5, .5), 0.))
        delta = torch.stack(steps, -1)
    xy = xy+delta
    present = present & (xy[..., 0] >= 0) & (xy[..., 0] <= w-1) & (xy[..., 1] >= 0) & (xy[..., 1] <= h-1)
    return {'xy':torch.where(present[..., None], xy, -1.),
            'scores':torch.where(present, confidence, 0.), 'valid':present,
            'truncated':peak.flatten(1).sum(-1) > k}
