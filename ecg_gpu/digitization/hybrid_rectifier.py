"""Image-only lattice refinement and geometry-only evaluation primitives.

The fit uses detected points, relative integer identities and confidence only.
Reference coordinates belong exclusively to the separate evaluation function.
No missing cell is filled and no global injectivity claim is made.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from .cell_mesh import forward_cell_map


def refine_lattice(xy, scores, topology, *, smoothness=.05, iterations=3):
    """Robust confidence-weighted fit with second differences of grid nodes.

    Linear second differences penalize local curvature rather than absolute
    position. This preserves affine perspective locally, but can oversmooth a
    real crease, so smoothing must be ablated on validation before deployment.
    """
    if not np.isfinite(smoothness) or smoothness < 0 or iterations < 1:
        raise ValueError('Nonnegative finite smoothness and positive iterations required')
    xy, scores = np.asarray(xy, float), np.asarray(scores, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or scores.shape != (len(xy),):
        raise ValueError('Matching point and confidence arrays required')
    if not np.isfinite(xy).all() or not np.isfinite(scores).all():
        raise ValueError('Finite observations required')
    selected = np.asarray(topology['selected_indices'], int)
    ids = np.asarray(topology['ids'], int)
    if len(selected) < 4 or ids.shape != (len(selected), 2):
        raise ValueError('At least four organized grid nodes required')
    observed = xy[selected]
    confidence = np.clip(scores[selected], .05, 1.)
    lookup = {tuple(k): i for i, k in enumerate(ids)}
    rows, columns, values = [], [], []
    count = 0
    for i, identity in enumerate(ids):
        for axis in (0, 1):
            before, after = identity.copy(), identity.copy()
            before[axis] -= 1
            after[axis] += 1
            if tuple(before) in lookup and tuple(after) in lookup:
                rows.extend([count]*3)
                columns.extend([lookup[tuple(before)], i, lookup[tuple(after)]])
                values.extend([1., -2., 1.])
                count += 1
    penalty = sparse.csr_matrix((values, (rows, columns)), shape=(count, len(ids)))
    fitted, weights = observed.copy(), confidence.copy()
    for _ in range(iterations):
        data = sparse.diags(np.sqrt(weights))
        matrix = sparse.vstack((data, np.sqrt(smoothness)*penalty)).tocsr()
        rhs = np.vstack((np.sqrt(weights)[:, None]*observed, np.zeros((count, 2))))
        fitted = np.column_stack([lsqr(matrix, rhs[:, a], atol=1e-8, btol=1e-8)[0] for a in (0, 1)])
        residual = np.linalg.norm(fitted-observed, axis=1)
        scale = max(.1, float(np.median(residual))*1.4826)
        weights = confidence*np.minimum(1., 1.5*scale/np.maximum(residual, 1e-8))
    # Reject smoothing locally if it reverses a previously supported cell.
    cells = []
    for r, c in topology['cells']:
        keys = ((r, c), (r, c+1), (r+1, c+1), (r+1, c))
        quad = fitted[[lookup[k] for k in keys]]
        edge = np.roll(quad, -1, axis=0)-quad
        following = np.roll(edge, -1, axis=0)
        cross = edge[:, 0]*following[:, 1]-edge[:, 1]*following[:, 0]
        if (cross > .01).all():
            cells.append([int(r), int(c)])
    return fitted, dict(topology, cells=cells,
        status='accepted' if len(cells) >= 16 else 'rejected',
        smoothing=smoothness, smoothing_displacement_p95_px=float(np.quantile(np.linalg.norm(fitted-observed, axis=1), .95)),
        global_injectivity_checked=False)


def forward_supported(observed, canonical, ids, query, cells):
    """Map evaluation points through supported triangles without extrapolation."""
    return forward_cell_map(observed, canonical, ids, query, cells)


def similarity_residual(predicted, reference):
    """Evaluate shape after one isotropic scale, translation and orthogonal map.

    A general affine fit would hide aspect-ratio and shear errors. Reflection
    is permitted because a relative lattice can swap or reverse its axes.
    The fitted alignment is evaluation-only, never an inference correction.
    """
    predicted, reference = np.asarray(predicted, float), np.asarray(reference, float)
    if predicted.shape != reference.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError('Matching [N,2] arrays required')
    keep = np.isfinite(predicted).all(1) & np.isfinite(reference).all(1)
    a, b = predicted[keep], reference[keep]
    result = {'reference_points': len(predicted), 'supported_points': int(keep.sum()),
              'point_support_fraction': float(keep.mean()) if len(keep) else 0.,
              'median_mm': None, 'p95_mm': None, 'rmse_mm': None}
    if len(a) < 4:
        return result
    a, b = a-a.mean(0), b-b.mean(0)
    if np.linalg.matrix_rank(a) < 2 or np.linalg.matrix_rank(b) < 2:
        return result
    u, singular, vt = np.linalg.svd(a.T@b)
    scale = singular.sum()/max(float((a*a).sum()), 1e-20)
    error = np.linalg.norm(scale*a@(u@vt)-b, axis=1)
    return dict(result, median_mm=float(np.median(error)), p95_mm=float(np.quantile(error, .95)),
                rmse_mm=float(np.sqrt(np.mean(error**2))))
