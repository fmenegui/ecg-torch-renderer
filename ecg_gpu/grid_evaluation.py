"""Offline point evaluation and diagnostic rectification, outside training.

SciPy is optional. No CPU conversion or matching belongs in a rendering loss.
Pixel coordinates always refer to pixel centers. Missing detections are counted.
"""
import math

import numpy as np


def _points(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a finite [N,2] array')
    return value


def match_grid_points(predicted, reference, *, radius=1.5):
    """Maximum-cardinality, then minimum-distance one-to-one radius matching.

    Dummy assignments cost more than every possible change in valid distances.
    Thus a close duplicate cannot count twice and greedy matching cannot reduce
    recall. Returned errors describe matched points only, alongside FP and FN.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    from scipy.spatial import cKDTree
    predicted, reference = _points(predicted, 'predicted'), _points(reference, 'reference')
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError('Matching radius must be positive and finite')
    n, m = len(predicted), len(reference)
    if max(n, m) > 65536:
        raise ValueError('Evaluate at most 65536 points per image or use separate regions')
    if n and m:
        predicted_tree, reference_tree = cKDTree(predicted), cKDTree(reference)
        edge_count = predicted_tree.count_neighbors(reference_tree, radius)
        if edge_count > 16_777_216:
            raise ValueError('Matching neighborhood too dense. Use smaller regions or radius')
        distance = predicted_tree.sparse_distance_matrix(reference_tree, radius, output_type='coo_matrix')
        penalty = (min(n, m)+1)*radius
        # One private unmatched edge per prediction avoids an N^2 dummy matrix.
        # Sparse matching treats explicit zeros as absent, so zero-distance
        # edges receive a constant positive epsilon. There are always n edges
        # in a full assignment, keeping the optimization objective unchanged.
        epsilon = np.finfo(float).eps
        cost = coo_matrix((np.r_[distance.data+epsilon, np.full(n, penalty+epsilon)],
            (np.r_[distance.row, np.arange(n)], np.r_[distance.col, m+np.arange(n)])), shape=(n, m+n)).tocsr()
        rows, columns = min_weight_full_bipartite_matching(cost)
        accepted = columns < m
        rows, columns = rows[accepted], columns[accepted]
        errors = np.linalg.norm(predicted[rows]-reference[columns], axis=-1)
    else:
        rows, columns = np.empty(0, dtype=int), np.empty(0, dtype=int)
        errors = np.empty(0)
    count = len(errors)
    return {'tp': count, 'fp': n-count, 'fn': m-count,
            'prediction_indices': rows, 'reference_indices': columns, 'errors_px': errors}


def summarize_grid_matches(matches):
    """Micro counts and conditional localization, with null undefined metrics."""
    matches = list(matches)
    tp, fp, fn = (sum(m[key] for m in matches) for key in ('tp', 'fp', 'fn'))
    errors = np.concatenate([np.asarray(m['errors_px']) for m in matches]) if matches else np.empty(0)
    return {'images': len(matches), 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'median_error_px': float(np.median(errors)) if len(errors) else None,
            'p95_error_px': float(np.quantile(errors, .95)) if len(errors) else None,
            'localization_population': 'matched points only'}


def rectification_diagnostic(observed, canonical, ids):
    """Held-out checkerboard-node interpolation with ORACLE lattice identities.

    This measures geometric usefulness after correct correspondence. It is not
    an inferred grid-ordering algorithm. Out-of-hull nodes remain unsupported.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError
    observed, canonical = _points(observed, 'observed'), _points(canonical, 'canonical')
    ids = _points(ids, 'ids')
    if observed.shape != canonical.shape or ids.shape != observed.shape:
        raise ValueError('Corresponding observed/canonical/ID arrays must have one shape')
    fit = ids.sum(-1).astype(np.int64) % 2 == 0
    result = {'oracle_correspondence': True, 'fit_nodes': int(fit.sum()),
              'held_out_nodes': int((~fit).sum()), 'supported_nodes': 0,
              'median_error_canonical_px': None, 'p95_error_canonical_px': None}
    if fit.sum() < 3 or not (~fit).any():
        return result
    try:
        estimated = LinearNDInterpolator(observed[fit], canonical[fit])(observed[~fit])
    except QhullError:
        return result
    supported = np.isfinite(estimated).all(-1)
    errors = np.linalg.norm(estimated[supported]-canonical[~fit][supported], axis=-1)
    if len(errors):
        result.update(supported_nodes=len(errors), median_error_canonical_px=float(np.median(errors)),
                      p95_error_canonical_px=float(np.quantile(errors, .95)))
    return result


def rectify_grid_image(image, observed, canonical, *, output_size):
    """Piecewise-linear inverse map from supplied ordered correspondences.

    No extrapolation outside their convex hull. The caller must resolve lattice
    identities and reject occlusions/foldovers. This is a diagnostic primitive,
    not a complete unwarping pipeline. Returns RGB and interpolation support.
    """
    import torch
    from torch.nn import functional as F
    from scipy.interpolate import LinearNDInterpolator
    observed, canonical = _points(observed, 'observed'), _points(canonical, 'canonical')
    if observed.shape != canonical.shape or len(observed) < 3:
        raise ValueError('At least three corresponding points required')
    if image.ndim != 4 or image.shape[:2] != (1, 3) or not image.is_floating_point():
        raise ValueError('image must be floating [1,3,H,W]')
    height, width = output_size
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError('Positive integer output size required')
    yy, xx = np.mgrid[:height, :width]
    mapping = LinearNDInterpolator(canonical, observed)(xx, yy)
    supported = np.isfinite(mapping).all(-1)
    supported &= ((mapping >= 0) & (mapping <= [image.shape[-1]-1, image.shape[-2]-1])).all(-1)
    mapping = np.nan_to_num(mapping, nan=-2.)
    field = torch.as_tensor(mapping, dtype=image.dtype, device=image.device)
    scale = image.new_tensor([image.shape[-1], image.shape[-2]])
    field = (field+.5)*2/scale-1
    support = torch.as_tensor(supported, device=image.device)[None, None]
    rectified = F.grid_sample(image, field[None], align_corners=False)*support
    return rectified, support
