"""Experimental soft localization refinement with fixed lattice identities.

This offline NumPy/SciPy stage is outside rendering and training hot paths.
Scores supply bounded ranking weights, never calibrated covariances. The
projective-residual curvature prior can bias genuine folds. Wrong lattice
identities and major/minor scale ambiguity are not resolved by this module.
"""
import numbers
import warnings

import numpy as np


def _rank_weights(scores):
    """Tie-invariant ranks in [0.25, 1], with equal scores giving unit weights."""
    from scipy.stats import rankdata

    if np.ptp(scores) == 0:
        return np.ones(len(scores))
    ranks = rankdata(scores, method='average')
    return .25+.75*(ranks-ranks.min())/(ranks.max()-ranks.min())


def _projective_prior(chart, observed, weights):
    """Normalized weighted linear fit, with bounded robust reweighting."""
    center_a, center_b = chart.mean(0), observed.mean(0)
    scale_a = np.sqrt(np.mean(np.sum((chart-center_a)**2, axis=1))/2)
    scale_b = np.sqrt(np.mean(np.sum((observed-center_b)**2, axis=1))/2)
    if min(scale_a, scale_b) <= 1e-10:
        return None
    a, b = (chart-center_a)/scale_a, (observed-center_b)/scale_b
    matrix = np.zeros((2*len(a), 8))
    homogeneous = np.column_stack((a, np.ones(len(a))))
    matrix[::2, :3], matrix[1::2, 3:6] = homogeneous, homogeneous
    matrix[::2, 6:], matrix[1::2, 6:] = -b[:, :1]*a, -b[:, 1:]*a
    working = weights.copy()
    prediction = None
    for _ in range(3):
        root = np.repeat(np.sqrt(working), 2)
        coefficients, _, rank, _ = np.linalg.lstsq(matrix*root[:, None], b.ravel()*root, rcond=None)
        if rank < 8 or not np.isfinite(coefficients).all():
            return None
        h = np.r_[coefficients, 1.].reshape(3, 3)
        value = homogeneous@h.T
        denominator = value[:, 2]
        if (np.min(np.abs(denominator)) < 1e-8
                or not ((denominator > 0).all() or (denominator < 0).all())):
            return None
        prediction = value[:, :2]/denominator[:, None]
        error = np.linalg.norm(prediction-b, axis=1)
        cutoff = max(2.5*np.median(error), 1e-8)
        working = weights*np.maximum(.05, np.minimum(1., cutoff/np.maximum(error, 1e-15)))
    return prediction*scale_b+center_b


def _stencils(ids):
    """Only adjacent observed IDs form constraints, so holes are not bridged."""
    from scipy.sparse import coo_matrix

    lookup = {tuple(identity): i for i, identity in enumerate(ids)}
    row, col, value, quads = [], [], [], []
    constraint = 0
    for i, (r, c) in enumerate(ids):
        for before, after in (((r-1, c), (r+1, c)), ((r, c-1), (r, c+1))):
            if before in lookup and after in lookup:
                row.extend([constraint]*3)
                col.extend([lookup[before], i, lookup[after]])
                value.extend([1., -2., 1.])
                constraint += 1
        corners = ((r, c), (r, c+1), (r+1, c+1), (r+1, c))
        if all(k in lookup for k in corners):
            quad = [lookup[k] for k in corners]
            quads.append(quad)
            row.extend([constraint]*4)
            col.extend(quad)
            value.extend(np.sqrt(2)*np.array([1., -1., 1., -1.]))
            constraint += 1
    operator = coo_matrix((value, (row, col)), shape=(constraint, len(ids))).tocsr()
    return operator, np.asarray(quads, int).reshape(-1, 4)


def _quad_cross(points, quads):
    vertices = points[quads]
    edges = np.roll(vertices, -1, axis=1)-vertices
    following = np.roll(edges, -1, axis=1)
    return edges[..., 0]*following[..., 1]-edges[..., 1]*following[..., 0]


def _coherent_curvature_guard(residual, ids, spacing):
    """Conservative prior-conflict test, not a true-crease classifier.

    A second difference repeated across four adjacent orthogonal grid lines
    can be physical shape or coherent detection error. Either conflicts with
    blindly shrinking curvature. No gold, missing-node interpolation or image
    texture enters this test. Oblique, weak or noisy bends can escape it.
    """
    lookup = {tuple(identity): i for i, identity in enumerate(ids)}
    families = []
    triggered = False
    for axis in (0, 1):
        groups = {}
        for i, identity in enumerate(ids):
            before, after = identity.copy(), identity.copy()
            before[axis] -= 1
            after[axis] += 1
            if tuple(before) in lookup and tuple(after) in lookup:
                value = residual[lookup[tuple(before)]]-2*residual[i]+residual[lookup[tuple(after)]]
                groups.setdefault(int(identity[axis]), {})[int(identity[1-axis])] = value
        deviations = []
        for group in groups.values():
            if len(group) >= 4:
                values = np.asarray(list(group.values()))
                deviations.extend(values-np.median(values, axis=0))
        # This is a robust tuning scale. It is not a calibrated uncertainty.
        noise = (float(np.median(np.linalg.norm(deviations, axis=1))/np.sqrt(2*np.log(2)))
                 if len(deviations) else 0.)
        noise = max(noise, .01*spacing)
        threshold = max(3*noise, .04*spacing)
        windows = []
        for group in groups.values():
            for start in sorted(group):
                if all(start+k in group for k in range(4)):
                    windows.append([group[start+k] for k in range(4)])
        coherent, peak = 0, 0.
        if windows:
            values = np.asarray(windows)
            center = np.median(values, axis=1)
            magnitude = np.linalg.norm(center, axis=1)
            dispersion = np.quantile(np.linalg.norm(values-center[:, None], axis=2), .75, axis=1)
            agreement = ((values*center[:, None]).sum(2) > 0).sum(1)
            selected = (magnitude > threshold) & (dispersion <= .5*magnitude) & (agreement >= 3)
            coherent = int(selected.sum())
            peak = float(magnitude[selected].max()) if coherent else 0.
        families.append({'difference_axis': axis, 'adjacent_lines_required': 4,
                         'windows_checked': len(windows), 'coherent_windows': coherent,
                         'robust_tuning_scale_px': noise,
                         'curvature_threshold_px': threshold,
                         'peak_coherent_second_difference_px': peak})
        triggered |= coherent > 0
    return bool(triggered), {'families': families,
                            'meaning': 'coherent curvature conflicts with smoothing prior',
                            'true_crease_certified': False,
                            'scope': 'four adjacent grid-line consensus, weak/noisy/oblique bends may escape'}


def refine_grid_points(xy, ids, scores=None, *, strength=.15, iterations=5,
                       max_shift_fraction=.2, max_points=8192, crease_guard=False):
    """Return refined pixel centers and diagnostics, preserving order and IDs.

    A weighted sparse solve smooths second differences of residuals from a
    fitted projective prior. IRLS reduces support for inconsistent locations.
    Every move is capped by observed axial spacing and a backtracking guard
    preserves the convex orientation of initially valid complete cells.
    The optional crease guard abstains on coherent curvature incompatible
    with this smoothing prior. It does not distinguish folds from coherent
    detector errors and does not promise to catch every physical fold.

    This does not certify physical accuracy or a globally injective output.
    Existing map construction must still validate the resulting geometry.
    """
    points, identities = np.asarray(xy, float), np.asarray(ids)
    if (points.ndim != 2 or points.shape[1:] != (2,) or identities.shape != points.shape
            or not np.isfinite(points).all() or not np.isfinite(identities).all()):
        raise ValueError('Finite corresponding [N,2] pixel centers and IDs required')
    if (not np.equal(identities, np.round(identities)).all()
            or np.any(np.abs(identities) > 2**30)
            or len(np.unique(identities, axis=0)) != len(identities)):
        raise ValueError('Unique bounded integer lattice IDs required')
    identities = identities.astype(np.int64)
    confidence = np.ones(len(points)) if scores is None else np.asarray(scores, float)
    if confidence.shape != (len(points),) or not np.isfinite(confidence).all():
        raise ValueError('Finite per-point ranking scores required')
    if not isinstance(strength, numbers.Real) or not np.isfinite(strength) or strength < 0:
        raise ValueError('Finite nonnegative refinement strength required')
    if (not isinstance(iterations, numbers.Integral) or isinstance(iterations, bool)
            or not 1 <= iterations <= 20):
        raise ValueError('Integer iterations in [1,20] required')
    if (not isinstance(max_shift_fraction, numbers.Real) or not np.isfinite(max_shift_fraction)
            or not 0 <= max_shift_fraction <= .5):
        raise ValueError('Finite maximum shift fraction in [0,0.5] required')
    if (not isinstance(max_points, numbers.Integral) or isinstance(max_points, bool)
            or not 4 <= max_points <= 65536):
        raise ValueError('Integer max_points in [4,65536] required')
    if type(crease_guard) is not bool:
        raise ValueError('Boolean crease_guard required')
    report = {'status': 'unchanged', 'reason': None, 'input_points': len(points),
              'method': 'sparse robust projective-residual curvature refinement',
              'fixed_lattice_ids': True, 'gold_used': False,
              'score_weights': 'bounded rank, not calibrated probability or covariance',
              'physical_accuracy_certified': False, 'global_injectivity_checked': False,
              'strength': float(strength), 'iteration_limit': int(iterations),
              'max_shift_fraction': float(max_shift_fraction), 'max_points': int(max_points),
              'iterations': 0, 'constraints': 0, 'cells_guarded': 0,
              'maximum_shift_px': 0., 'maximum_shift_fraction': 0.,
              'converged': None, 'iteration_limit_reached': False,
              'convergence_scope': 'IRLS residual iterates before shift cap and cell guard',
              'crease_guard_enabled': crease_guard, 'crease_guard_triggered': False,
              'budget_exhausted': False}

    def unchanged(reason):
        return points.copy(), dict(report, reason=reason)

    if len(points) > max_points:
        report['budget_exhausted'] = True
        return unchanged('point budget exceeded')
    if strength == 0 or max_shift_fraction == 0:
        return unchanged('refinement disabled')
    if len(points) < 9 or min(len(np.unique(identities[:, a])) for a in (0, 1)) < 3:
        return unchanged('insufficient 2D lattice support')
    operator, quads = _stencils(identities)
    report['constraints'] = operator.shape[0]
    if operator.shape[0] < 4:
        return unchanged('insufficient adjacent-ID constraints')
    from scipy.spatial import cKDTree
    from scipy.sparse import diags
    from scipy.sparse.linalg import MatrixRankWarning, spsolve

    distances = cKDTree(points).query(points, k=2)[0][:, 1]
    spacing = float(np.median(distances))
    if spacing <= 1e-8 or np.any(distances <= 1e-10):
        return unchanged('degenerate observed spacing')
    # Axial neighbors are preferred for the displacement cap. Fallback to a
    # global median only when an isolated node lacks any adjacent observed ID.
    lookup = {tuple(identity): i for i, identity in enumerate(identities)}
    local_spacing = np.full(len(points), spacing)
    for i, (r, c) in enumerate(identities):
        neighbors = [lookup[k] for k in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)) if k in lookup]
        if neighbors:
            local_spacing[i] = np.median(np.linalg.norm(points[neighbors]-points[i], axis=1))
    quality = _rank_weights(confidence)
    prior = _projective_prior(identities[:, ::-1].astype(float), points, quality)
    if prior is None:
        return unchanged('no finite nondegenerate projective prior')
    residual = points-prior
    if np.max(np.abs(residual)) < 1e-9*max(spacing, 1):
        return unchanged('projective observations already consistent')
    if crease_guard:
        triggered, diagnostic = _coherent_curvature_guard(residual, identities, spacing)
        report.update(crease_guard_triggered=triggered, curvature_guard=diagnostic)
        if triggered:
            return unchanged('coherent curvature conflicts with smoothing prior')
    penalty = strength*(operator.T@operator)
    weights, previous, refined_residual = quality.copy(), residual, residual
    cutoff = 0.
    converged = False
    for iteration in range(iterations):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', MatrixRankWarning)
                refined_residual = spsolve(diags(weights)+penalty, weights[:, None]*residual)
        except (MatrixRankWarning, RuntimeError, ValueError) as error:
            return unchanged('sparse solve failed: '+type(error).__name__)
        if not np.isfinite(refined_residual).all():
            return unchanged('nonfinite sparse solution')
        innovation = residual-refined_residual
        centered = innovation-np.median(innovation, axis=0)
        sigma = 1.4826*float(np.median(np.abs(centered)))
        cutoff = max(2.5*sigma, .01*spacing)
        magnitude = np.linalg.norm(innovation, axis=1)
        robust = np.maximum(.05, np.minimum(1., cutoff/np.maximum(magnitude, 1e-15)))
        weights = quality*robust
        report['iterations'] = iteration+1
        if np.max(np.linalg.norm(refined_residual-previous, axis=1)) <= 1e-5*spacing:
            converged = True
            break
        previous = refined_residual
    report.update(converged=converged, iteration_limit_reached=not converged,
                  budget_exhausted=not converged)
    shift = prior+refined_residual-points
    norm = np.linalg.norm(shift, axis=1)
    cap = max_shift_fraction*local_spacing
    shift *= np.minimum(1., cap/np.maximum(norm, 1e-15))[:, None]
    original_cross = _quad_cross(points, quads)
    valid = ((original_cross > 1e-8).all(1) | (original_cross < -1e-8).all(1))
    guarded = quads[valid]
    original_cross = original_cross[valid]
    report['cells_guarded'] = len(guarded)
    factor = 1.
    for _ in range(8):
        candidate = points+factor*shift
        current_cross = _quad_cross(candidate, guarded)
        if np.all(current_cross*original_cross > 0) and np.all(np.abs(current_cross) >= .2*np.abs(original_cross)):
            break
        factor *= .5
    else:
        return unchanged('cell orientation guard rejected refinement')
    final_shift = np.linalg.norm(candidate-points, axis=1)
    report.update(status='refined', reason=None, backtracking_factor=factor,
                  maximum_shift_px=float(final_shift.max()),
                  maximum_shift_fraction=float(np.max(final_shift/local_spacing)),
                  median_shift_px=float(np.median(final_shift)),
                  score_weight_min=float(quality.min()), score_weight_max=float(quality.max()),
                  robust_downweighted_points=int((weights < .5*quality).sum()),
                  robust_cutoff_px=float(cutoff),
                  clipped_points=int((norm > cap).sum()),
                  invalid_original_cells_unguarded=int((~valid).sum()))
    return candidate, report
