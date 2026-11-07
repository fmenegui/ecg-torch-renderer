"""Independent analytical conformance tests of the actual M3 metric core."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT/'examples/rectification_comparison_metrics.py'
spec = importlib.util.spec_from_file_location('production_m3_metric_core', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def fixture(transform=lambda x: x.copy()):
    x, y = np.meshgrid(np.arange(8)*2., np.arange(8)*2.)
    anchor = np.column_stack((x.ravel(), y.ravel()))
    grid = np.array([[1., 1.], [3., 1.], [1., 3.], [3., 3.]])
    paper = np.array([[20., 20.], [30., 20.], [20., 30.], [30., 30.]])
    queries = {d:{'reference':r} for d, r in [('anchors',anchor), ('grid',grid), ('paper',paper)]}
    predictions = {d:{'coords':transform(v['reference']),
                      'claimed_eligible':np.ones(len(v['reference']),bool)} for d,v in queries.items()}
    return queries, predictions


class MetricCoreChecks(unittest.TestCase):
    def test_exact_and_threshold_boundary(self):
        ref = np.zeros((4, 2))
        pred = np.array([[0.,0.],[.3,.4],[.50001,0.],[3.,4.]])
        r = core.score_coordinates(ref,pred,np.ones(4,bool))
        self.assertEqual((r['g'],r['w'],r['c'],r['r']),(.5,.5,1.,.5))
        self.assertEqual(r['queries'],4)

    def test_claimed_nan_and_infinity_are_wrong(self):
        ref = np.zeros((4,2))
        pred = np.array([[0.,0.],[np.nan,0.],[0.,np.inf],[0.,0.]])
        r = core.score_coordinates(ref,pred,np.ones(4,bool))
        self.assertEqual((r['g'],r['w'],r['c'],r['r']),(.5,.5,1.,.5))
        self.assertEqual(r['claimed_nonfinite_queries'],2)
        self.assertEqual(r['finite_claimed_queries'],2)
        self.assertEqual(r['rmse_mm'],0.)

    def test_abstain_all_never_has_defined_zero_risk(self):
        r = core.score_coordinates(np.zeros((4,2)),np.full((4,2),np.nan),np.zeros(4,bool))
        self.assertEqual((r['g'],r['w'],r['c']),(0.,0.,0.))
        self.assertIsNone(r['r'])
        self.assertIsNone(r['rmse_mm'])

    def test_rejection_cannot_improve_correct_coverage(self):
        ref = np.zeros((4,2))
        pred = np.array([[0.,0.],[0.,0.],[3.,4.],[3.,4.]])
        full = core.score_coordinates(ref,pred,np.ones(4,bool))
        partial = core.score_coordinates(ref,pred,[True,True,False,False])
        self.assertEqual(full['g'],partial['g'])
        self.assertEqual((partial['w'],partial['c'],partial['r']),(0.,.5,0.))

    def test_translation_removed_once_but_raw_is_retained(self):
        q,p = fixture(lambda x:x+[31.,-11.])
        r = core.score_predictions(p,q)
        np.testing.assert_array_equal(r['primary']['registration']['offset_mm'],[-31.,11.])
        for d in ('grid','paper'):
            self.assertEqual(r['primary']['domains'][d]['g'],1.)
            self.assertEqual(r['raw_native_origin']['domains'][d]['g'],0.)

    def test_offset_is_not_refit_for_paper(self):
        q,p = fixture(lambda x:x+[31.,-11.])
        p['paper']['coords'] += [2.,0.]
        r = core.score_predictions(p,q)
        self.assertEqual(r['primary']['domains']['grid']['g'],1.)
        self.assertEqual(r['primary']['domains']['paper']['w'],1.)

    def test_fivefold_scale_is_not_rescued(self):
        q,p = fixture(lambda x:5*x)
        r = core.score_predictions(p,q)
        for track in ('primary','secondary_D4'):
            self.assertEqual(r[track]['domains']['grid']['g'],0.)
            self.assertEqual(r[track]['domains']['grid']['w'],1.)

    def test_reflection_is_failure_primary_and_only_diagnostic_rescue(self):
        q,p = fixture(lambda x:x*[-1.,1.])
        r = core.score_predictions(p,q)
        self.assertEqual(r['primary']['domains']['grid']['g'],0.)
        self.assertEqual(r['secondary_D4']['domains']['grid']['g'],1.)
        self.assertNotEqual(r['secondary_D4']['registration']['D4_index'],0)

    def test_D4_selected_from_anchors_not_scoring_queries(self):
        q,p = fixture(lambda x:x*[-1.,1.])
        # Identity with anchor-derived +14 x origin is perfect on scoring queries,
        # but reflected anchors require the reflection for the diagnostic choice.
        for d in ('grid','paper'):
            p[d]['coords'] = q[d]['reference']+[-14.,0.]
        r = core.score_predictions(p,q)
        self.assertEqual(r['primary']['domains']['grid']['g'],1.)
        self.assertEqual(r['secondary_D4']['domains']['grid']['g'],0.)

    def test_insufficient_anchors_retains_native_coverage(self):
        q,p = fixture()
        p['anchors']['claimed_eligible'][2:] = False
        r = core.score_predictions(p,q)
        self.assertEqual(r['primary']['registration']['status'],'registration_unavailable')
        for d in ('grid','paper'):
            self.assertEqual(r['primary']['domains'][d]['g'],0.)
            self.assertEqual(r['primary']['domains'][d]['c'],0.)
            self.assertEqual(r['primary']['domains'][d]['native_claimed_coverage'],1.)
            self.assertIsNone(r['primary']['domains'][d]['r'])

    def test_degenerate_reference_span_fails_registration(self):
        q,p = fixture()
        q['anchors']['reference'][:,0] = 0.
        p['anchors']['coords'][:,0] = 0.
        r = core.score_predictions(p,q)
        self.assertEqual(r['primary']['registration']['status'],'registration_unavailable')

    def test_collapsed_prediction_does_not_escape_wrong_mass(self):
        q,p = fixture(lambda x:np.zeros_like(x))
        r = core.score_predictions(p,q)
        self.assertEqual(r['primary']['registration']['status'],'registered_translation_only')
        self.assertEqual(r['primary']['domains']['grid']['w'],1.)
        self.assertEqual(r['primary']['domains']['grid']['c'],1.)
        self.assertEqual(r['primary']['registration']['predicted_span_mm'],[0.,0.])

    def test_claimed_nan_after_registration_remains_wrong(self):
        q,p = fixture()
        p['grid']['coords'][0] = np.nan
        r = core.score_predictions(p,q)
        self.assertEqual((r['primary']['domains']['grid']['g'],r['primary']['domains']['grid']['w']),(.75,.25))
        self.assertEqual(r['primary']['domains']['grid']['c'],1.)

    def test_unrestricted_field_does_not_infer_footprint_from_confidence(self):
        q,p = fixture()
        r = core.score_predictions(p,q)
        self.assertEqual(r['unrestricted_finite_field']['status'],'not_available_missing_native_footprint')

    def test_full_RGB_finite_field_and_confidence_abstention_are_separate(self):
        q,p = fixture(lambda x:x+[31.,-11.])
        for d in p:
            p[d]['claimed_eligible'][:] = False
            p[d]['native_footprint'] = np.ones(len(p[d]['coords']),bool)
        r = core.score_predictions(p,q)
        self.assertEqual(r['primary']['domains']['paper']['c'],0.)
        self.assertEqual(r['unrestricted_finite_field']['domains']['paper']['g'],1.)
        self.assertEqual(r['unrestricted_finite_field']['registration']['offset_source'],
                         'unrestricted_finite_footprint_anchors_after_primary_registration_unavailable')

    def test_crop_footprint_is_retained_and_nonfinite_footprint_disclosed(self):
        q,p = fixture()
        for d in p:
            p[d]['native_footprint'] = np.ones(len(p[d]['coords']),bool)
        p['paper']['native_footprint'][2:] = False
        p['paper']['coords'][0] = np.nan
        r = core.score_predictions(p,q)
        u = r['unrestricted_finite_field']['domains']['paper']
        self.assertEqual(u['c'],.25)
        self.assertEqual(u['native_RGB_footprint_coverage'],.5)
        self.assertEqual(u['nonfinite_inside_native_RGB_footprint'],1)

    def test_prepare_queries_deterministic_and_disjoint(self):
        y,x = np.mgrid[:24,:30]
        material = np.stack((x*.2,y*.3),axis=-1)
        bank = {'reference_mm':material,'valid':np.ones((24,30))}
        a = np.array([[1.,1.],[3.,1.],[1.,3.],[3.,3.]])
        b = np.array([[3.,3.],[12.,12.],[15.,16.],[20.,21.]])
        cached = {'grid_query':a,'paper_query':b,
                  'grid_reference':material[a[:,1].astype(int),a[:,0].astype(int)],
                  'paper_reference':material[b[:,1].astype(int),b[:,0].astype(int)]}
        r = core.prepare_case_queries(bank,cached,'independent-toy')
        repeat = core.prepare_case_queries(bank,cached,'independent-toy')
        np.testing.assert_array_equal(r['anchors']['source'],repeat['anchors']['source'])
        anchors = {tuple(v) for v in r['anchors']['source']}
        self.assertEqual(len(anchors),64)
        self.assertFalse(anchors & ({tuple(v) for v in a}|{tuple(v) for v in b}))
        actual_hash = hashlib.sha256(np.asarray(r['anchors']['source'],dtype='<f8').tobytes()).hexdigest()
        self.assertEqual(actual_hash,r['anchors']['source_sha256'])
        broken = copy.deepcopy(cached)
        broken['grid_reference'][0,0] += 1.
        with self.assertRaises(ValueError):
            core.prepare_case_queries(bank,broken,'independent-toy')

    def test_invalid_domain_and_conflicting_masks_fail(self):
        with self.assertRaises(ValueError):
            core.score_coordinates(np.empty((0,2)),np.empty((0,2)),np.empty(0,bool))
        with self.assertRaises(ValueError):
            core.score_coordinates(np.full((1,2),np.nan),np.zeros((1,2)),[True])
        with self.assertRaises(ValueError):
            core.score_coordinates(np.zeros((1,2)),np.zeros((1,2)),[True],threshold_mm=0)
        q,p = fixture()
        p['grid']['eligible'] = np.zeros(4,bool)
        with self.assertRaises(ValueError):
            core.score_predictions(p,q)

    def test_aggregation_uses_patients_and_ratio_of_means(self):
        rows = [
            {'group_id':'A','backend':'x','g':1.,'w':0.,'c':1.} for _ in range(9)
        ]+[{'group_id':'B','backend':'x','g':0.,'w':.5,'c':.5}]
        r = core.aggregate_patient_backend(rows)['overall']
        self.assertEqual(r['g'],.5)
        self.assertEqual(r['w'],.25)
        self.assertEqual(r['c'],.75)
        self.assertAlmostEqual(r['r'],1/3)

    def test_constant_paired_effect_and_missing_cases(self):
        rows = [{'key':f'{p}-{b}-{v}','group_id':p,'backend':b,
                 'g':.3,'w':.3,'c':.6}
                for p in ('A','B') for b in ('x','y') for v in (0,1)]
        improved = [dict(r,g=.4,w=.2) for r in rows]
        r = core.paired_patient_bootstrap({'B0':rows,'other':improved},'B0',repetitions=100)
        effect = r['methods']['other']['delta_95_descriptive_interval']
        for end in ('lower','upper'):
            self.assertAlmostEqual(effect['g'][end],.1)
            self.assertAlmostEqual(effect['w'][end],-.1)
            self.assertAlmostEqual(effect['r'][end],-1/6)
        with self.assertRaises(ValueError):
            core.paired_patient_bootstrap({'B0':rows,'other':improved[1:]},'B0',repetitions=100)
        altered = copy.deepcopy(improved)
        altered[0]['key'] = 'replacement-after-failure'
        with self.assertRaises(ValueError):
            core.paired_patient_bootstrap({'B0':rows,'other':altered},'B0',repetitions=100)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MetricCoreChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'status':'pass' if result.wasSuccessful() else 'fail',
               'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
               'production_scorer_exercised':True,'source_file':str(CORE),
               'source_sha256':hashlib.sha256(CORE.read_bytes()).hexdigest(),
               'scope':'Analytic scoring, origin/D4 separation, footprint, patient pairing and query-contract checks only'}
    Path(__file__).with_name('metric-core-review.json').write_text(json.dumps(receipt,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
