"""Input-only experimental geometry selection. Never consumes scoring truth.

Canonical IDs and pitch are assumed. No numerical test here verifies those
assumptions, absolute orientation, or physical correctness.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.linalg import solve
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
CELL_PATH = WORKSPACE/'ecg-torch-renderer/ecg_gpu/digitization/cell_mesh.py'
CELL_SHA = '566253fba786de84475f00d593b80876b3869499d3703c18a71c9a5396cf37d6'
PROTOCOL_SHA = 'a3ae872d70b34206a8ff809bca10046aa1d3adc81b6c9d9765f50506ac544fe4'
if hashlib.sha256(CELL_PATH.read_bytes()).hexdigest() != CELL_SHA:
    raise RuntimeError('Frozen M2 cell implementation changed')
if hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest() != PROTOCOL_SHA:
    raise RuntimeError('Frozen M4 geometry protocol changed')
spec = importlib.util.spec_from_file_location('m4_frozen_cells', CELL_PATH)
cells = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cells)
PROTOCOL = json.loads((ROOT/'protocol.json').read_text())
CANDIDATES = PROTOCOL['candidate_order']


def project(points, h):
    points = np.asarray(points, float)
    q = np.c_[points, np.ones(len(points))]@h.T
    valid = np.isfinite(q).all(1) & (np.abs(q[:, 2]) > 1e-12)
    out = np.full(points.shape, np.nan)
    out[valid] = q[valid, :2]/q[valid, 2:]
    return out


def fit_h(observed, canonical, estimator):
    if len(observed) < 4 or np.linalg.matrix_rank(np.c_[observed, np.ones(len(observed))]) < 3:
        raise ValueError('insufficient noncollinear anchors')
    cv2.setRNGSeed(61023)
    h, inliers = cv2.findHomography(observed, canonical,
        0 if estimator == 'LS' else cv2.RANSAC, 1.5,
        maxIters=2000, confidence=.995)
    if h is None or not np.isfinite(h).all():
        raise ValueError('no finite homography')
    s = np.linalg.svd(h, compute_uv=False)
    if s[-1] <= 1e-12*s[0]:
        raise ValueError('singular homography')
    guard_h(h, observed)
    return h, {'estimator': estimator, 'inlier_fraction': float(inliers.mean())}


def guard_h(h, observed):
    d = np.c_[observed, np.ones(len(observed))]@h[2]
    if np.min(np.abs(d)) <= 1e-10*max(np.max(np.abs(d)), 1) or not ((d > 0).all() or (d < 0).all()):
        raise ValueError('projective horizon crosses observed source hull')


def lattice(ids):
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.shape[1] != 2 or not np.isfinite(ids).all() or not np.equal(ids, np.round(ids)).all():
        raise ValueError('finite integer [N,2] IDs required')
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError('duplicate integer IDs')
    low, high = ids.min(0).astype(int), ids.max(0).astype(int)
    nr, nc = high-low+1
    if nr*nc > 2000 or min(nr, nc) < 2:
        raise ValueError('unsupported rectangular ID lattice extent')
    rr, cc = np.mgrid[low[0]:high[0]+1, low[1]:high[1]+1]
    all_ids = np.c_[rr.ravel(), cc.ravel()]
    index = (ids[:, 0].astype(int)-low[0])*nc+ids[:, 1].astype(int)-low[1]
    rows = []
    def add(indices, weights):
        row = np.zeros(nr*nc)
        row[np.asarray(indices)] = weights
        rows.append(row)
    for r in range(nr):
        for c in range(nc):
            i = r*nc+c
            if c+2 < nc:
                add([i, i+1, i+2], [1, -2, 1])
            if r+2 < nr:
                add([i, i+nc, i+2*nc], [1, -2, 1])
            if c+1 < nc and r+1 < nr:
                add([i, i+1, i+nc, i+nc+1], np.sqrt(2)*np.array([1, -1, -1, 1]))
    d = np.asarray(rows)
    return all_ids, index, d.T@d


def fit_field(residual, selected_indices, n_vertices, penalty, strength):
    """Fixed ID basis, so withheld source coordinates never define its knots."""
    w = np.ones(len(selected_indices))
    value = np.zeros((n_vertices, 2))
    for _ in range(4):
        diagonal = np.full(n_vertices, 1e-8)
        diagonal[selected_indices] += w
        rhs = np.zeros_like(value)
        rhs[selected_indices] = residual*w[:, None]
        a = strength*penalty + np.diag(diagonal)
        value = solve(a, rhs, assume_a='pos', check_finite=False)
        error = np.linalg.norm(value[selected_indices]-residual, axis=1)
        w = np.minimum(1., 1./np.maximum(error, 1e-12))
    return value


def parse_candidate(name):
    estimator, tail = name.split('/')
    if tail == 'global':
        return estimator, None, 0.
    tokens = tail.split('-')
    return estimator, float(tokens[1]), float(tokens[3])


def guard_complex(source, triangles, label):
    """Conservative local, overlap and boundary-contact exclusion checks.

    This is a bounded computational check, not a physical certificate. A closed
    strip can have consistent local orientation and zero positive-area overlap
    while its distinct boundary points coincide. The boundary checks reject it.
    """
    vertices = source[triangles]
    edge_a = vertices[:, 1]-vertices[:, 0]
    edge_b = vertices[:, 2]-vertices[:, 0]
    cross = edge_a[:, 0]*edge_b[:, 1]-edge_a[:, 1]*edge_b[:, 0]
    scale = np.max(np.sum((vertices[:, 1]-vertices[:, 0])**2, axis=1))
    tolerance = max(scale*1e-10, 1e-12)
    if np.any(np.abs(cross) <= tolerance):
        raise ValueError(label+': degenerate triangle')
    if not ((cross > 0).all() or (cross < 0).all()):
        raise ValueError(label+': mixed triangle orientation/fold')
    referenced = np.unique(triangles)
    length_tolerance = max(np.sqrt(scale)*1e-9, 1e-10)
    if cKDTree(source[referenced]).query_pairs(length_tolerance):
        raise ValueError(label+': coincident nonincident vertices/non-simple boundary')
    edge_counts = {}
    for t in triangles:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = tuple(sorted((int(a), int(b))))
            edge_counts[key] = edge_counts.get(key, 0)+1
    if any(count > 2 for count in edge_counts.values()):
        raise ValueError(label+': nonmanifold triangle edge')
    boundary = np.asarray([edge for edge, count in edge_counts.items() if count == 1])
    segments = source[boundary]
    b_lower, b_upper = segments.min(1), segments.max(1)
    def orient(a, b, c):
        ab, ac = b-a, c-a
        return ab[0]*ac[1]-ab[1]*ac[0]
    for i in range(len(boundary)):
        candidates = np.flatnonzero(((b_upper[i]+length_tolerance >= b_lower[i+1:])
            & (b_upper[i+1:]+length_tolerance >= b_lower[i])).all(1))+i+1
        for j in candidates:
            if len(set(boundary[i]) & set(boundary[j])):
                continue
            a, b = segments[i]
            c, d = segments[j]
            eps = length_tolerance*max(np.linalg.norm(b-a), np.linalg.norm(d-c), 1.)
            o1, o2 = orient(a, b, c), orient(a, b, d)
            o3, o4 = orient(c, d, a), orient(c, d, b)
            straddle_ab = min(o1, o2) <= eps and max(o1, o2) >= -eps
            straddle_cd = min(o3, o4) <= eps and max(o3, o4) >= -eps
            if straddle_ab and straddle_cd:
                raise ValueError(label+': touching/crossing nonincident boundary edges')
    lower, upper = vertices.min(1), vertices.max(1)
    for i in range(len(triangles)):
        candidates = np.flatnonzero(((upper[i] > lower[i+1:]+1e-8) & (upper[i+1:] > lower[i]+1e-8)).all(1))+i+1
        for j in candidates:
            # Exact shared edges can only overlap after a local orientation flip,
            # which was rejected above. Other pairs need an explicit area test.
            if len(set(triangles[i]) & set(triangles[j])) >= 2:
                continue
            area, _ = cv2.intersectConvexConvex(vertices[i].astype(np.float32), vertices[j].astype(np.float32))
            if area > max(tolerance, 1e-5):
                raise ValueError(label+': positive-area overlap/ambiguous inverse')
    return {'triangles': len(triangles), 'orientation': int(np.sign(cross[0])),
            'positive_area_overlaps': 0, 'nonincident_boundary_contacts': 0,
            'coincident_distinct_vertices': 0, 'guard_revision': 2}


class Geometry:
    """One shared continuous residual complex with conservative partial support."""
    def __init__(self, observed, canonical, ids, allowed_cells, h, field_source=None):
        self.observed = np.asarray(observed, float)
        self.canonical = np.asarray(canonical, float)
        self.h = h
        guard_h(h, observed)
        self.h_inverse = np.linalg.inv(h)
        self.triangles = cells.cell_triangles(observed, canonical, ids, allowed_cells)
        if not len(self.triangles):
            raise ValueError('no valid original observed cells')
        self.source_report = guard_complex(observed, self.triangles, 'observed complex')
        self.support = cells.CellMap(observed, observed, self.triangles)
        self.residual = field_source is not None
        self.field_source = project(observed, h) if field_source is None else field_source
        if self.residual:
            self.field_report = guard_complex(self.field_source, self.triangles, 'fitted normalized complex')
            guard_complex(canonical, self.triangles, 'target complex')
            self.local = cells.CellMap(self.field_source, canonical, self.triangles)
            self.local_inverse = cells.CellMap(canonical, self.field_source, self.triangles)
        else:
            self.field_report = None

    def forward(self, query, return_status=False):
        query = np.asarray(query, float)
        answer = np.full(query.shape, np.nan)
        status = np.full(len(query), 'outside-observed-support', dtype='<U48')
        valid = np.isfinite(query).all(1)
        ix = np.flatnonzero(valid)
        if len(ix):
            valid[ix] = np.isfinite(self.support(query[ix])).all(1)
        ix = np.flatnonzero(valid)
        if len(ix):
            z = project(query[ix], self.h)
            value = self.local(z) if self.residual else z
            good = np.isfinite(value).all(1)
            answer[ix[good]] = value[good]
            status[ix] = 'outside-fitted-support'
            status[ix[good]] = 'returned-with-uncalibrated-uncertainty'
        return (answer, status) if return_status else answer

    def inverse(self, physical, return_status=False):
        physical = np.asarray(physical, float)
        normalized = self.local_inverse(physical) if self.residual else physical.copy()
        answer = np.full(physical.shape, np.nan)
        finite = np.isfinite(normalized).all(1)
        ix = np.flatnonzero(finite)
        if len(ix):
            raw = project(normalized[ix], self.h_inverse)
            good = np.isfinite(raw).all(1)
            if good.any():
                back = self.forward(raw[good])
                consistent = np.isfinite(back).all(1) & (np.linalg.norm(back-physical[ix[good]], axis=1) < 1e-5)
                answer[ix[good][consistent]] = raw[good][consistent]
        status = np.where(np.isfinite(answer).all(1), 'returned-with-uncalibrated-uncertainty', 'ambiguous-or-inconsistent')
        return (answer, status) if return_status else answer


def fit_select(observed, ids, allowed_cells, pitch_mm=5.):
    """Only points and supplied IDs enter selection. No truth argument exists."""
    started = time.perf_counter()
    observed, ids = np.asarray(observed, float), np.asarray(ids)
    if observed.shape != ids.shape or not np.isfinite(observed).all():
        raise ValueError('matching finite [N,2] observations required')
    all_ids, index, penalty = lattice(ids)
    canonical = ids[:, ::-1]*pitch_mm
    all_canonical = all_ids[:, ::-1]*pitch_mm
    folds = ((ids[:, 0]//3)+2*(ids[:, 1]//4)) % 4
    records = {name: {'fold_loss': [], 'fold_rmse_mm': [], 'fold_status': []} for name in CANDIDATES}
    for fold in range(4):
        held = folds == fold
        train = ~held
        if not held.any():
            raise ValueError('empty spatial validation fold')
        for estimator in ('LS', 'RANSAC'):
            try:
                h, _ = fit_h(observed[train], canonical[train], estimator)
                z_train = project(observed[train], h)
                z_held = project(observed[held], h)
                if not np.isfinite(z_held).all():
                    raise ValueError('validation crossed horizon')
                values = {None: np.zeros_like(all_canonical)}
                for strength in (30., 3., .3, .03):
                    values[strength] = fit_field(z_train-canonical[train], index[train], len(all_ids), penalty, strength)
                for name in CANDIDATES:
                    e, strength, shrink = parse_candidate(name)
                    if e != estimator:
                        continue
                    prediction = canonical[held]+shrink*values[strength][index[held]]
                    error = np.linalg.norm(prediction-z_held, axis=1)
                    records[name]['fold_loss'].append(float(np.mean(np.minimum(error, 2.)**2)))
                    records[name]['fold_rmse_mm'].append(float(np.sqrt(np.mean(error**2))))
                    records[name]['fold_status'].append('fit')
            except (ValueError, np.linalg.LinAlgError) as error:
                for name in CANDIDATES:
                    if parse_candidate(name)[0] == estimator:
                        records[name]['fold_loss'].append(4.)
                        records[name]['fold_rmse_mm'].append(None)
                        records[name]['fold_status'].append(str(error))
    for record in records.values():
        losses = np.asarray(record['fold_loss'])
        record['mean_loss'] = float(losses.mean())
        record['standard_error'] = float(losses.std(ddof=1)/np.sqrt(4))
        record['full_fit_status'] = 'not fitted'
    best = min(CANDIDATES, key=lambda name: records[name]['mean_loss'])
    threshold = records[best]['mean_loss']+records[best]['standard_error']
    selected = next(name for name in CANDIDATES if records[name]['mean_loss'] <= threshold+1e-14)
    models = {}
    for estimator in ('LS', 'RANSAC'):
        try:
            h, h_report = fit_h(observed, canonical, estimator)
            z = project(observed, h)
            fields = {None: np.zeros_like(all_canonical)}
            for strength in (30., 3., .3, .03):
                fields[strength] = fit_field(z-canonical, index, len(all_ids), penalty, strength)
            for name in CANDIDATES:
                e, strength, shrink = parse_candidate(name)
                if e != estimator:
                    continue
                try:
                    field_source = None if strength is None else canonical+shrink*fields[strength][index]
                    models[name] = Geometry(observed, canonical, ids, allowed_cells, h, field_source)
                    records[name]['full_fit_status'] = 'accepted'
                    records[name]['homography_fit'] = h_report
                    records[name]['geometry_guards'] = models[name].source_report
                except (ValueError, np.linalg.LinAlgError) as error:
                    records[name]['full_fit_status'] = 'rejected: '+str(error)
        except (ValueError, np.linalg.LinAlgError) as error:
            for name in CANDIDATES:
                if parse_candidate(name)[0] == estimator:
                    records[name]['full_fit_status'] = 'rejected: '+str(error)
    model = models.get(selected)
    all_folds_failed = all(all(status != 'fit' for status in record['fold_status']) for record in records.values())
    if all_folds_failed:
        model = None
    report = {
        'selected_candidate': selected,
        'best_cv_candidate': best,
        'one_standard_error_threshold': threshold,
        'selected_geometry_status': 'accepted' if model is not None else 'rejected',
        'all_cv_folds_failed': all_folds_failed,
        'selection_used_physical_reference': False,
        'cv_fold_assignment': folds.astype(int).tolist(),
        'candidates': records,
        'identity_status': 'supplied IDs, pitch, origin and orientation remain unverified',
        'uncertainty_status': 'uncalibrated predictive diagnostics, not a physical error bound',
        'selected_fold_rmse_mm': records[selected]['fold_rmse_mm'],
        'selection_seconds_cpu': time.perf_counter()-started,
    }
    return model, report, models
