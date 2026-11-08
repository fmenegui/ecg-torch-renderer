"""Checks of saved results and guard-v2 regression without physical tuning."""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import geometry_selector as geom
from run_experiment import forward_reference

ROOT = Path(__file__).resolve().parent


def run():
    original = json.loads((ROOT/'results/results.json').read_text())
    revised = json.loads((ROOT/'results-guard-v2/results.json').read_text())
    differences, changed_reasons = [], []
    selected_same, candidates_same, arrays_same, scores_same = True, True, True, True
    for a, b in zip(original['cases'], revised['cases']):
        name = a['scene']['name']
        assert name == b['scene']['name']
        selected_same &= a['selection']['selected_candidate'] == b['selection']['selected_candidate']
        for candidate in geom.CANDIDATES:
            before = a['selection']['candidates'][candidate]['full_fit_status']
            after = b['selection']['candidates'][candidate]['full_fit_status']
            candidates_same &= (before == 'accepted') == (after == 'accepted')
            if before != after:
                changed_reasons.append({'scene': name, 'candidate': candidate,
                    'before': before, 'after': after})
        with np.load(ROOT/'results'/a['array_file']) as za, np.load(ROOT/'results-guard-v2'/b['array_file']) as zb:
            for key in za.files:
                identical = np.array_equal(za[key], zb[key], equal_nan=True) if za[key].dtype.kind in 'fc' else np.array_equal(za[key], zb[key])
                if not identical:
                    differences.append({'scene': name, 'array': key})
                    arrays_same = False
        for method in geom.PROTOCOL['comparators']:
            for metric in ['G','W','C','R','rmse_mm','p95_mm','p99_mm','max_mm','common_rmse_mm','direct_common_rmse_mm','exterior_returned']:
                scores_same &= a['methods'][method][metric] == b['methods'][method][metric]
    controls = {}
    collinear = np.c_[np.arange(16), np.zeros(16)]
    try:
        geom.fit_h(collinear, collinear, 'LS')
        controls['collinear_rejected'] = False
    except ValueError:
        controls['collinear_rejected'] = True
    twisted = np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
    target = np.array([[0.,0.],[5.,0.],[0.,5.],[5.,5.]])
    try:
        geom.fit_h(twisted, target, 'LS')
        controls['horizon_rejected'] = False
    except ValueError as error:
        controls['horizon_rejected'] = 'horizon' in str(error)
    overlap = np.array([[0.,0.],[2.,0.],[0.,2.],[.2,.2],[1.8,.2],[.2,1.8]])
    triangles = np.array([[0,1,2],[3,4,5]])
    try:
        geom.guard_complex(overlap, triangles, 'overlap control')
        controls['positive_area_overlap_rejected'] = False
    except ValueError:
        controls['positive_area_overlap_rejected'] = True
    values = geom.cells.CellMap(overlap, overlap, triangles)(np.array([[.4,.4]]))
    controls['multiple_interior_inverse_returns_missing'] = bool(np.isnan(values).all())
    theta = np.linspace(0, 2*np.pi, 25)
    rr, cc = np.mgrid[:2,:25]
    ids = np.c_[rr.ravel(),cc.ravel()]
    source = np.concatenate([radius*np.c_[np.cos(theta),np.sin(theta)] for radius in [1.,2.]])
    canonical = ids[:, ::-1]*5.
    triangles = geom.cells.cell_triangles(source, canonical, ids)
    try:
        geom.guard_complex(source, triangles, 'annulus control')
        controls['nonincident_boundary_contact_rejected'] = False
    except ValueError as error:
        controls['nonincident_boundary_contact_rejected'] = 'nonincident' in str(error)
        controls['annulus_rejection_reason'] = str(error)
    # Test strict exterior of the actual observed source domain, independently
    # of whether noisy observations happen to lie outside physical paper.
    strict_exterior_returned = 0
    for case in revised['cases']:
        z = np.load(ROOT/'results-guard-v2'/case['array_file'])
        selected, report, models = geom.fit_select(z['observed'], z['ids'], z['cells'])
        low, high = z['observed'].min(0), z['observed'].max(0)
        span = np.maximum(high-low, 1.)
        query = np.array([low-span, high+span, low-[span[0],0], high+[0,span[1]]])
        if selected is not None:
            strict_exterior_returned += int(np.isfinite(selected.forward(query)).all(1).sum())
    controls['strict_observed_exterior_returned'] = strict_exterior_returned
    controls['strict_observed_exterior_all_missing'] = strict_exterior_returned == 0
    # A no-noise central gap control is generated independently of physical scores.
    rr, cc = np.mgrid[:12,:16]
    ids = np.c_[rr.ravel(),cc.ravel()]
    keep = ~((ids[:,0] >= 4)&(ids[:,0] <= 7)&(ids[:,1] >= 6)&(ids[:,1] <= 9))
    ids = ids[keep]
    allowed = np.array([[r,c] for r in range(11) for c in range(15)])
    scene = {'kind':'flat','k':.02}
    observed = forward_reference(ids[:,::-1]*5., scene)
    selected, _, _ = geom.fit_select(observed, ids, allowed)
    center = forward_reference(np.array([[37.5,27.5]]), scene)
    controls['central_missing_cells_not_bridged'] = bool(selected is not None and np.isnan(selected.forward(center)).all())
    controls['selector_accepts_only_observations_ids_cells_pitch'] = list(inspect.signature(geom.fit_select).parameters) == ['observed','ids','allowed_cells','pitch_mm']
    controls['projective_plane_precision_mm'] = max(c['methods']['selected-input-only']['rmse_mm'] for c in revised['cases'] if c['scene']['name'] in ['planar-affine','planar-projective','planar-projective-strong'])
    controls['projective_plane_precision_pass'] = controls['projective_plane_precision_mm'] < 2e-7
    controls['seam_max_mm'] = max(c['methods']['selected-input-only']['seams']['forward_seam_max_mm'] for c in revised['cases'] if c['methods']['selected-input-only']['seams'])
    controls['seam_pass'] = controls['seam_max_mm'] < 1e-7
    controls['decision_seal_unchanged'] = revised['decision_seal_unchanged']
    controls['coherent_pitch_alias_retained_as_failure'] = next(c['methods']['selected-input-only']['W'] for c in revised['cases'] if c['scene']['name']=='coherent-pitch-alias') == 1.
    result = {'schema':'m4-guard-v2-same-experiment-retest',
        'old_geometry_source_sha256': original['source_sha256']['m4/geometry/geometry_selector.py'],
        'new_geometry_source_sha256': revised['source_sha256']['m4/geometry/geometry_selector.py'],
        'guard_revision_sha256': hashlib.sha256((ROOT/'guard-revision-v2.json').read_bytes()).hexdigest(),
        'same_selected_candidate_all_19': bool(selected_same),
        'same_full_fit_accepted_rejected_all_342': bool(candidates_same),
        'same_saved_arrays_including_all_candidates': bool(arrays_same),
        'same_scores_all_76_arms': bool(scores_same),
        'array_differences': differences, 'changed_rejection_reasons': changed_reasons,
        'controls': controls,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result['pass'] = bool(selected_same and candidates_same and arrays_same and scores_same
        and all(value for key,value in controls.items() if isinstance(value,bool)))
    (ROOT/'guard-v2-retest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    run()
