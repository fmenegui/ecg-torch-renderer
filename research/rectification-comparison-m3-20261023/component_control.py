"""Independent interpolation-primitive ablation, not a commercial reproduction.

No learned points, grid inference, extrapolation, smoothing or RGB resampling.
The same noisy-or-exact corners and IDs feed each geometric primitive.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PITCH = 5.0
METHODS = ('triangles', 'homography', 'bilinear')
CAMERA = np.array([[16., 2., 3., 20.], [1., 13., -6., 30.],
                   [.035, .017, .11, 1.]])
SCENES = [
    {'key': 'plane-exact', 'surface': 'plane', 'noise_sigma_px': 0., 'seed': 1973},
    {'key': 'cylinder-axis', 'surface': 'cylinder', 'angle': 0., 'radius_mm': 10.,
     'noise_sigma_px': 0., 'seed': 1973},
    {'key': 'cylinder-oblique', 'surface': 'cylinder', 'angle': .47,
     'radius_mm': 10., 'noise_sigma_px': 0., 'seed': 1973},
    {'key': 'hinge-aligned', 'surface': 'hinge', 'angle': 0., 'hinge_radians': .55,
     'center_mm': 5., 'noise_sigma_px': 0., 'seed': 1973},
    {'key': 'hinge-within-cell', 'surface': 'hinge', 'angle': 0.,
     'hinge_radians': .55, 'center_mm': 2.5, 'noise_sigma_px': 0., 'seed': 1973},
    {'key': 'hinge-oblique', 'surface': 'hinge', 'angle': .47,
     'hinge_radians': .55, 'center_mm': 5., 'noise_sigma_px': 0., 'seed': 1973},
    *[{'key': f'plane-noisy-{s}', 'surface': 'plane', 'noise_sigma_px': .35,
       'seed': s} for s in (1973, 7919, 104729)],
    {'key': 'invalid-observed-cells', 'surface': 'plane', 'noise_sigma_px': 0.,
     'seed': 1973, 'corrupt_center': True},
]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def material_embedding(xy, scene):
    """Isometric plane/cylinder/hinge embedding of the flat material sheet."""
    xy = np.asarray(xy, float)
    angle = scene.get('angle', 0.)
    c, s = np.cos(angle), np.sin(angle)
    center = scene.get('center_mm', 5.)
    d = xy-np.array([center, 5.])
    along = d[:, 0]*c+d[:, 1]*s
    transverse = -d[:, 0]*s+d[:, 1]*c
    z = np.zeros(len(xy))
    embedded = along.copy()
    if scene['surface'] == 'cylinder':
        radius = scene['radius_mm']
        embedded = radius*np.sin(along/radius)
        z = radius*(1.-np.cos(along/radius))
    elif scene['surface'] == 'hinge':
        theta = scene['hinge_radians']
        embedded = along*np.cos(theta)
        z = np.abs(along)*np.sin(theta)
    return np.column_stack((center+embedded*c-transverse*s,
                            5.+embedded*s+transverse*c, z))


def observe(xy, scene):
    surface = material_embedding(xy, scene)
    value = np.column_stack((surface, np.ones(len(surface))))@CAMERA.T
    assert (value[:, 2] > 0).all()
    return value[:, :2]/value[:, 2:]


def homogeneous(xy, matrix):
    v = np.column_stack((xy, np.ones(len(xy))))@matrix.T
    ok = np.abs(v[:, 2]) > 1e-10
    result = np.full((len(xy), 2), np.nan)
    result[ok] = v[ok, :2]/v[ok, 2:]
    return result


def fit_homography(corners):
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    a = np.zeros((8, 8))
    one = np.column_stack((uv, np.ones(4)))
    a[::2, :3], a[1::2, 3:6] = one, one
    a[::2, 6:], a[1::2, 6:] = -corners[:, :1]*uv, -corners[:, 1:]*uv
    values = np.linalg.solve(a, corners.ravel())
    return np.r_[values, 1.].reshape(3, 3)


def bilinear_forward(uv, corners):
    u, v = uv[:, :1], uv[:, 1:]
    return ((1-u)*(1-v)*corners[0]+u*(1-v)*corners[1]
            + u*v*corners[2]+(1-u)*v*corners[3])


def forward(uv, corners, method):
    if method == 'homography':
        return homogeneous(uv, fit_homography(corners))
    if method == 'bilinear':
        return bilinear_forward(uv, corners)
    result = np.empty_like(uv)
    lower = uv[:, 0] >= uv[:, 1]
    u, v = uv[lower, :1], uv[lower, 1:]
    result[lower] = (1-u)*corners[0]+(u-v)*corners[1]+v*corners[2]
    u, v = uv[~lower, :1], uv[~lower, 1:]
    result[~lower] = (1-v)*corners[0]+u*corners[2]+(v-u)*corners[3]
    return result


def inverse(query, corners, method):
    if method == 'homography':
        uv = homogeneous(query, np.linalg.inv(fit_homography(corners)))
    elif method == 'bilinear':
        a, b = corners[1]-corners[0], corners[3]-corners[0]
        mixed = corners[0]-corners[1]+corners[2]-corners[3]
        uv = (query-corners[0])@np.linalg.inv(np.column_stack((a, b))).T
        for _ in range(30):
            residual = bilinear_forward(uv, corners)-query
            j = np.stack((a+uv[:, 1:]*mixed, b+uv[:, :1]*mixed), axis=2)
            determinant = np.linalg.det(j)
            alive = np.isfinite(determinant) & (np.abs(determinant) > 1e-10)
            step = np.zeros_like(uv)
            step[alive] = np.linalg.solve(j[alive], residual[alive, :, None])[..., 0]
            uv[~alive] = np.nan
            uv[alive] -= step[alive]
            if not alive.any() or np.max(np.linalg.norm(step[alive], axis=1)) < 1e-11:
                break
        residual = np.linalg.norm(bilinear_forward(uv, corners)-query, axis=1)
        uv[~(residual < 1e-7)] = np.nan
    else:
        uv = np.full((len(query), 2), np.nan)
        for indices, template in [([0, 1, 2], [[0., 0.], [1., 0.], [1., 1.]]),
                                  ([0, 2, 3], [[0., 0.], [1., 1.], [0., 1.]])]:
            tri = corners[indices]
            bary = (query-tri[0])@np.linalg.inv((tri[1:]-tri[0]).T).T
            weights = np.column_stack((1-bary.sum(1), bary))
            inside = (weights >= -1e-9).all(1)
            pick = inside & ~np.isfinite(uv[:, 0])
            uv[pick] = weights[pick]@np.asarray(template)
    inside = np.isfinite(uv).all(1) & (uv >= -1e-9).all(1) & (uv <= 1+1e-9).all(1)
    return uv, inside


def cell_corner_sets(nodes):
    result = []
    for r in range(2):
        for c in range(2):
            q = nodes[[r, r, r+1, r+1], [c, c+1, c+1, c]]
            result.append((r, c, q))
    return result


def cell_validity(corners):
    edge = np.roll(corners, -1, axis=0)-corners
    after = np.roll(edge, -1, axis=0)
    cross = edge[:, 0]*after[:, 1]-edge[:, 1]*after[:, 0]
    # For a bilinear quad, corner Jacobian signs bound the interior sign.
    det_tri = [np.linalg.det((corners[[1, 2]]-corners[0]).T),
               np.linalg.det((corners[[2, 3]]-corners[0]).T)]
    return bool((cross > 1e-8).all()), float(np.min(cross)), min(det_tri)


def predict(nodes, query, method):
    prediction = np.full((len(query), 2), np.nan)
    support = np.zeros(len(query), dtype=int)
    invalid, min_cross, min_tri = 0, [], []
    for r, c, corners in cell_corner_sets(nodes):
        valid, cross, tri = cell_validity(corners)
        min_cross.append(cross)
        min_tri.append(tri)
        if not valid:
            invalid += 1
            continue
        uv, eligible = inverse(query, corners, method)
        pick = eligible & (support == 0)
        prediction[pick] = (uv[pick]+np.array([c, r]))*PITCH
        support += eligible
    return prediction, support > 0, {'invalid_cells': invalid,
        'minimum_corner_jacobian_px2': min(min_cross),
        'minimum_triangle_determinant_px2': min(min_tri),
        'queries_with_multiple_cell_support': int((support > 1).sum()),
        'nonfinite_predictions': int((~np.isfinite(prediction).all(1)).sum())}


def metric(prediction, eligible, truth):
    error = np.linalg.norm(prediction-truth, axis=1)
    accepted = eligible & np.isfinite(error)
    g = float((accepted & (error <= .5)).mean())
    w = float((accepted & (error > .5)).mean())
    c = float(accepted.mean())
    e = error[accepted]
    return {'queries': len(truth), 'accepted': int(accepted.sum()),
            'G': g, 'W': w, 'C': c, 'R': w/c if c else None,
            'rmse_mm_on_returned': float(np.sqrt(np.mean(e**2))) if len(e) else None,
            'p95_mm_on_returned': float(np.quantile(e, .95)) if len(e) else None,
            'max_mm_on_returned': float(np.max(e)) if len(e) else None}


def seams(nodes, method):
    t = np.linspace(0., 1., 101)
    cells = {(r, c): q for r, c, q in cell_corner_sets(nodes)}
    values = []
    for r in range(2):
        qa, qb = cells[r, 0], cells[r, 1]
        if cell_validity(qa)[0] and cell_validity(qb)[0]:
            a = forward(np.column_stack((np.ones_like(t), t)), qa, method)
            b = forward(np.column_stack((np.zeros_like(t), t)), qb, method)
            values.extend(np.linalg.norm(a-b, axis=1))
    for c in range(2):
        qa, qb = cells[0, c], cells[1, c]
        if cell_validity(qa)[0] and cell_validity(qb)[0]:
            a = forward(np.column_stack((t, np.ones_like(t))), qa, method)
            b = forward(np.column_stack((t, np.zeros_like(t))), qb, method)
            values.extend(np.linalg.norm(a-b, axis=1))
    return {'samples': len(values), 'max_source_disagreement_px': float(max(values)) if values else None,
            'p95_source_disagreement_px': float(np.quantile(values, .95)) if values else None}


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'name': 'geometric primitive ablation, not commercial Dotter/Gridder',
        'date': '2026-10-23', 'scene_specs': SCENES, 'camera_matrix': CAMERA.tolist(),
        'material_grid_mm': [0., 5., 10.], 'query_seed': 40001,
        'independent_uniform_interior_queries': 12288, 'correct_threshold_mm': .5,
        'methods': list(METHODS), 'gold_used_to_select_model': False,
        'no_detection_no_topology_inference': True, 'all_methods_same_observed_corners': True,
        'out_of_cell_queries': 'ineligible, retain in fixed denominator',
        'shared_edge_tie': 'first row-major eligible cell, report multiple support',
        'analytical_scope': 'physical isometric plane/cylinder/hinge, ideal pinhole, four observed cells',
        'not_measured': ['commercial pipeline', 'RGB', 'waveform', 'extrapolation', 'clinical accuracy']}
    dump(out/'protocol.json', protocol)
    rng = np.random.default_rng(protocol['query_seed'])
    interior = rng.uniform(0., 10., (protocol['independent_uniform_interior_queries'], 2))
    t = np.linspace(0., 10., 41)
    boundary = np.unique(np.vstack([np.column_stack((np.full_like(t, u), t))
                                    for u in (0., 5., 10.)]+[
                                    np.column_stack((t, np.full_like(t, u)))
                                    for u in (0., 5., 10.)]), axis=0)
    canonical = np.array([[(x, y) for x in (0., 5., 10.)] for y in (0., 5., 10.)])
    results = []
    for scene in SCENES:
        observed = observe(canonical.reshape(-1, 2), scene).reshape(3, 3, 2)
        observed += np.random.default_rng(scene['seed']).normal(0., scene['noise_sigma_px'], observed.shape)
        if scene.get('corrupt_center'):
            observed[1, 1] += [150., -130.]
        source = observe(interior, scene)
        edges = observe(boundary, scene)
        arrays = {'material_queries_mm': interior, 'source_queries_px': source,
                  'material_edge_queries_mm': boundary, 'source_edge_queries_px': edges,
                  'observed_corners_px': observed, 'corner_material_mm': canonical}
        row = {'scene': scene, 'methods': {}}
        for method in METHODS:
            pred, eligible, validity = predict(observed, source, method)
            edge_pred, edge_eligible, edge_validity = predict(observed, edges, method)
            changed_nodes = observed*np.array([.63, 1.27])+[.5, -.5]
            changed_query = source*np.array([.63, 1.27])+[.5, -.5]
            changed, changed_eligible, _ = predict(changed_nodes, changed_query, method)
            common = eligible & changed_eligible
            parity = {'affine_coordinate_change': 'scale [.63,1.27], pixel-center translation [.5,-.5]',
                'eligibility_mismatches': int((eligible != changed_eligible).sum()),
                'max_prediction_difference_mm': float(np.max(np.linalg.norm(pred[common]-changed[common], axis=1))) if common.any() else None}
            row['methods'][method] = {'interior': metric(pred, eligible, interior),
                'boundary': metric(edge_pred, edge_eligible, boundary), 'validity': validity,
                'boundary_validity': edge_validity, 'shared_edge_seams': seams(observed, method),
                'coordinate_convention_control': parity}
            arrays[method+'_prediction_mm'] = pred
            arrays[method+'_eligible'] = eligible
            arrays[method+'_boundary_prediction_mm'] = edge_pred
            arrays[method+'_boundary_eligible'] = edge_eligible
        path = out/(scene['key']+'.npz')
        np.savez_compressed(path, **arrays)
        row.update(arrays_file=path.name, arrays_sha256=digest(path))
        results.append(row)
    report = {'protocol_sha256': digest(out/'protocol.json'), 'source_sha256': digest(Path(__file__)),
              'cases': results, 'complete': True, 'parameter_selection': False}
    dump(out/'results.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    output = run(args.out)
    print(json.dumps({'cases': len(output['cases']), 'methods': list(METHODS), 'complete': output['complete']}))
