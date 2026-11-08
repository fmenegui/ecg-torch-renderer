"""Predeclared geometry-only probe of a shared projective residual composition.

Not a Powerful Medical reproduction. No neural inference, image bank, RGB output
or parameter selection. The reference physical coordinates enter scoring only.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
CELL_SHA = '566253fba786de84475f00d593b80876b3869499d3703c18a71c9a5396cf37d6'


def setup(directory):
    if hashlib.sha256((directory/'cell_mesh.py').read_bytes()).hexdigest() != CELL_SHA:
        raise ValueError('The frozen M2 cell implementation changed')
    package = types.ModuleType('composition_audit')
    package.__path__ = [str(directory)]
    sys.modules['composition_audit'] = package
    return importlib.import_module('composition_audit.cell_mesh')


def project(points, h):
    value = np.column_stack((points, np.ones(len(points))))@h.T
    denominator = value[:, 2]
    good = np.isfinite(value).all(1) & (np.abs(denominator) > 1e-12)
    output = np.full(points.shape, np.nan)
    output[good] = value[good, :2]/denominator[good, None]
    return output


def homography(observed, canonical, estimator):
    cv2.setRNGSeed(1973)
    method = 0 if estimator == 'LS' else cv2.RANSAC
    h, inliers = cv2.findHomography(observed, canonical, method, 1.5,
                                   maxIters=2000, confidence=.995)
    report = {'estimator': estimator, 'inlier_fraction': None,
              'status': 'rejected', 'reason': None}
    if h is None or not np.isfinite(h).all():
        return None, dict(report, reason='no finite homography')
    report['inlier_fraction'] = float(inliers.mean()) if inliers is not None else None
    singular = np.linalg.svd(h, compute_uv=False)
    if singular[-1] <= 1e-12*singular[0]:
        return None, dict(report, reason='singular or numerically degenerate homography')
    denominator = np.column_stack((observed, np.ones(len(observed))))@h[2]
    if (np.min(np.abs(denominator)) <= 1e-10*max(np.max(np.abs(denominator)), 1.)
            or not ((denominator > 0).all() or (denominator < 0).all())):
        return None, dict(report, reason='projective horizon crosses observed source hull',
                          denominator_min=float(denominator.min()),
                          denominator_max=float(denominator.max()))
    return h, dict(report, status='accepted')


class Geometry:
    """Partial observed-cell geometry, with an optional global normalization."""
    def __init__(self, cells, observed, canonical, ids, allowed, *, h=None, residual=True):
        self.observed, self.canonical, self.h = observed, canonical, h
        self.residual = residual
        self.triangles = cells.cell_triangles(observed, canonical, ids, allowed)
        if not len(self.triangles):
            raise ValueError('no valid observed cell triangles')
        self.support = cells.CellMap(observed, observed, self.triangles)
        self.normalized = observed if h is None else project(observed, h)
        self.h_inverse = np.eye(3) if h is None else np.linalg.inv(h)
        if h is not None:
            other = cells.cell_triangles(self.normalized, canonical, ids, allowed)
            if not np.array_equal(other, self.triangles):
                raise ValueError('normalization changed valid cell topology')
        self.local = cells.CellMap(self.normalized, canonical, self.triangles)
        self.local_inverse = cells.CellMap(canonical, self.normalized, self.triangles)

    def forward(self, query):
        finite = np.isfinite(self.support(query)).all(1)
        output = np.full(query.shape, np.nan)
        if finite.any():
            value = query[finite] if self.h is None else project(query[finite], self.h)
            output[finite] = self.local(value) if self.residual else value
        return output

    def inverse(self, physical):
        value = self.local_inverse(physical) if self.residual else physical.copy()
        finite = np.isfinite(value).all(1)
        result = np.full(physical.shape, np.nan)
        if finite.any():
            result[finite] = project(value[finite], self.h_inverse)
        return result


def analytic(value, specification):
    p = value.copy()
    kind = specification['kind']
    if kind == 'smooth':
        amplitudes = specification['amplitudes_mm']
        p[:, 0] += amplitudes[0]*np.sin(2*np.pi*value[:, 1]/55)
        p[:, 1] += amplitudes[1]*np.sin(2*np.pi*value[:, 0]/75)
    elif kind == 'aligned':
        p[:, 1] += specification['slope']*np.abs(value[:, 0]-specification['crease_u_mm'])
    elif kind == 'oblique':
        hinge = value[:, 1]-.6*value[:, 0]-specification['line_v_minus_0_6u_mm']
        p[:, 0] += specification['slope']*np.maximum(hinge, 0)
    elif kind != 'flat':
        raise ValueError(kind)
    return p*10/(1+specification['k']*p[:, :1])


def metrics(prediction, truth):
    finite = np.isfinite(prediction).all(1)
    error = np.linalg.norm(prediction-truth, axis=1)
    good = finite & (error <= .5)
    bad = finite & (error > .5)
    e = error[finite]
    return {'queries': len(truth), 'accepted_queries': int(finite.sum()),
            'G': float(good.mean()), 'W': float(bad.mean()), 'C': float(finite.mean()),
            'R': float(bad.sum()/finite.sum()) if finite.any() else None,
            'rmse_mm': float(np.sqrt(np.mean(e**2))) if len(e) else None,
            'p95_mm': float(np.quantile(e, .95)) if len(e) else None,
            'p99_mm': float(np.quantile(e, .99)) if len(e) else None,
            'max_mm': float(e.max()) if len(e) else None}, finite, error


def triangle_value(query, source, target):
    a = np.stack((source[0]-source[2], source[1]-source[2]), axis=1)
    first = np.linalg.solve(a, query-source[2])
    return np.r_[first, 1-first.sum()]@target


def seams(geometry):
    if not geometry.residual:
        return {'shared_edges': None, 'forward_seam_max_mm': 0., 'inverse_seam_max_px': 0.,
                'reason': 'One global homography has no per-cell seam'}
    adjacency = {}
    for index, t in enumerate(geometry.triangles):
        for a,b in ((t[0],t[1]), (t[1],t[2]), (t[2],t[0])):
            adjacency.setdefault(tuple(sorted((int(a),int(b)))), []).append(index)
    forward, reverse = [], []
    for edge, adjacent in adjacency.items():
        if len(adjacent) != 2:
            continue
        source_query = geometry.observed[list(edge)].mean(0)[None]
        normalized_query = source_query if geometry.h is None else project(source_query, geometry.h)
        canonical_query = geometry.canonical[list(edge)].mean(0)
        values, backs = [], []
        for index in adjacent:
            t = geometry.triangles[index]
            values.append(triangle_value(normalized_query[0], geometry.normalized[t], geometry.canonical[t]))
            b = triangle_value(canonical_query, geometry.canonical[t], geometry.normalized[t])
            backs.append(project(b[None], geometry.h_inverse)[0])
        forward.append(np.linalg.norm(values[0]-values[1]))
        reverse.append(np.linalg.norm(backs[0]-backs[1]))
    return {'shared_edges': len(forward),
            'forward_seam_max_mm': float(max(forward, default=0)),
            'inverse_seam_max_px': float(max(reverse, default=0)),
            'scope': 'C0 shared-edge agreement, not C1 tangent continuity or physical accuracy'}


def run(geometry_dir, out):
    protocol_path = ROOT/'composition-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    cells = setup(geometry_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = out/'arrays'
    arrays.mkdir(exist_ok=True)
    r,c = np.mgrid[:12,:16]
    ids = np.column_stack((r.ravel(),c.ravel()))
    canonical = ids[:, ::-1]*5.
    allowed = [[row,col] for row in range(11) for col in range(15)]
    rng = np.random.default_rng(protocol['queries']['seed'])
    queries = rng.uniform([.001,.001],[74.999,54.999],(4096,2))
    cases = []
    for specification in protocol['cases']:
        observed = analytic(canonical,specification)
        if specification['noise_px']:
            random = np.random.default_rng(specification['seed'])
            observed += random.normal(0,specification['noise_px'],observed.shape)
        if specification.get('anchor_fault','').startswith('x shift 5*'):
            observed[:,0] += 5*np.sin(2*np.pi*ids[:,0]/11)
        if specification.get('anchor_fault','').startswith('x shift 13'):
            observed[canonical[:,0] >= 25,0] += 13
        source_queries = analytic(queries,specification)
        methods, predictions, homographies = {}, {}, {}
        mappings = {}
        try:
            mappings['direct-cells'] = Geometry(cells,observed,canonical,ids,allowed)
        except (ValueError,np.linalg.LinAlgError) as error:
            methods['direct-cells'] = {'rejected':str(error)}
        for estimator in ('LS','RANSAC'):
            h, report = homography(observed,canonical,estimator)
            homographies[estimator] = report
            for composition in (False,True):
                name = ('composition-' if composition else 'global-H-')+estimator
                if h is None:
                    methods[name] = {'rejected':report['reason']}
                    continue
                try:
                    mappings[name] = Geometry(cells,observed,canonical,ids,allowed,h=h,residual=composition)
                except (ValueError,np.linalg.LinAlgError) as error:
                    methods[name] = {'rejected':str(error)}
        saved = {'physical_queries':queries,'source_queries':source_queries,
                 'observed':observed,'canonical':canonical,'ids':ids,'cells':np.asarray(allowed)}
        for name in ('direct-cells','global-H-LS','composition-LS','global-H-RANSAC','composition-RANSAC'):
            mapping = mappings.get(name)
            prediction = mapping.forward(source_queries) if mapping is not None else np.full(queries.shape,np.nan)
            result, finite, error = metrics(prediction,queries)
            predictions[name] = (finite,error)
            saved[name+'_predicted'] = prediction
            result.update(methods.get(name,{}))
            result['seams'] = seams(mapping) if mapping is not None else None
            if mapping is not None and finite.any():
                back = mapping.inverse(prediction[finite])
                good = np.isfinite(back).all(1)
                result['roundtrip_queries'] = int(good.sum())
                result['roundtrip_max_px'] = float(np.linalg.norm(back[good]-source_queries[finite][good],axis=1).max()) if good.any() else None
            methods[name] = result
        base_finite,base_error = predictions['direct-cells']
        for name,result in methods.items():
            finite,error = predictions[name]
            common = finite & base_finite
            result['common_with_direct_queries'] = int(common.sum())
            result['common_rmse_mm'] = float(np.sqrt(np.mean(error[common]**2))) if common.any() else None
            result['direct_common_rmse_mm'] = float(np.sqrt(np.mean(base_error[common]**2))) if common.any() else None
        np.savez_compressed(arrays/(specification['name']+'.npz'),**saved)
        cases.append({'case':specification,'homography_fit':homographies,'methods':methods,
                      'array_file':'arrays/'+specification['name']+'.npz'})
    # A unit fault in correspondence ordering forces a finite projective fit
    # with a horizon through the source cell. It is not a native image test.
    twisted = np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
    corner_ids = np.array([[0,0],[0,1],[1,0],[1,1]])
    h,twisted_report = homography(twisted,corner_ids[:,::-1]*5.,'LS')
    assert h is None and 'horizon' in twisted_report['reason']
    collinear = np.column_stack((np.arange(16,dtype=float),np.zeros(16)))
    h,collinear_report = homography(collinear,canonical[:16],'LS')
    assert h is None
    for case in cases:
        for name, result in case['methods'].items():
            if result['seams'] is not None:
                assert result['seams']['forward_seam_max_mm'] < 1e-7
                assert result['seams']['inverse_seam_max_px'] < 1e-6
    result = {'protocol_sha256':protocol_sha,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cell_mesh_sha256':CELL_SHA,'opencv':cv2.__version__,
              'claim':'Analytical development probe only. No bank, detector, RGB delivery or proprietary product execution.',
              'cases':cases,'guard_faults':{'horizon':twisted_report,'collinear':collinear_report},
              'complete':True,'configuration_selected_from_scores':False}
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'composition-results')
    args = parser.parse_args()
    result = run(args.geometry_dir,args.out)
    print(json.dumps({'cases':len(result['cases']),'arms':sum(len(c['methods']) for c in result['cases']),
                      'complete':result['complete'],'out':str(args.out)}))
