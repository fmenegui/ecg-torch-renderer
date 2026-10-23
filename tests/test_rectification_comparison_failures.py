"""Inject failures into the actual orchestrator without neural inference."""
import ast
from contextlib import contextmanager, redirect_stdout
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'examples/compare_rectification_methods.py'
CORE = ROOT/'examples/rectification_comparison_metrics.py'
spec = importlib.util.spec_from_file_location('qa_orchestration_metric', CORE)
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(npj_factory):
    selected = {'digest', 'plain', 'write_json', 'deadline', 'verify_lock',
                'execute', 'failed_predictions', 'load_queries'}
    nodes = [n for n in ast.parse(SOURCE.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in selected]
    modules = {'metric': metric, 'npj': SimpleNamespace(NPJPerspectiveRectifier=npj_factory),
               'kaggle': SimpleNamespace()}
    namespace = {'Path': Path, 'hashlib': hashlib, 'json': json, 'np': np,
                 'time': time, 'signal': signal, 'contextmanager': contextmanager,
                 'Image': Image, 'gc': gc,
                 'torch': SimpleNamespace(Tensor=type('UnusedTensor', (), {}), set_num_threads=lambda n: None),
                 'cv2': SimpleNamespace(setNumThreads=lambda n: None, setRNGSeed=lambda n: None),
                 'METHODS': ('npj-generic',), 'DOMAINS': ('anchors', 'grid', 'paper'),
                 'load_module': lambda name, path: modules[Path(path).name]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


def fixture(temporary, pixel_budget=16000000):
    base = Path(temporary)
    out, bank = base/'out', base/'bank'
    (out/'queries').mkdir(parents=True)
    bank.mkdir()
    rows = []
    query_hashes = {}
    for view in range(2):
        key = 'analytic-'+str(view)
        np.savez_compressed(bank/(key+'.npz'), image=np.zeros((3, 3, 3), np.uint8))
        coordinates = np.array([[0., 0.], [2., 0.], [0., 2.], [2., 2.]])
        np.savez_compressed(out/'queries'/(key+'.npz'), **{
            d+'_'+kind: coordinates for d in ('anchors', 'grid', 'paper')
            for kind in ('source', 'reference')})
        query_hashes[key+'.npz'] = digest(out/'queries'/(key+'.npz'))
        rows.append({'key': key, 'group_id': 'analytic-patient', 'split': 'validation',
                     'backend': ('torch', 'offline')[view], 'layout': 'analytic', 'view': view,
                     'file': key+'.npz', 'sha256': digest(bank/(key+'.npz'))})
    (out/'schedule.json').write_text(json.dumps({'cases': rows}))
    lock = {'runtime_paths': {}, 'source_sha256': {}, 'weight_sha256': {},
            'owned_file_sha256': {'schedule.json': digest(out/'schedule.json')},
            'query_sha256': query_hashes, 'case_timeout_seconds': 180,
            'RGB_output_budget_pixels': pixel_budget}
    (out/'lock.json').write_text(json.dumps(lock))
    args = SimpleNamespace(out=out, bank=bank, metric_core=Path('metric'), npj_adapter=Path('npj'),
                           kaggle_adapter=Path('kaggle'), method='npj-generic', npj_repo=Path('unused'),
                           split='all', limit=0, no_resume=False)
    return args, rows


class OrchestrationFailureChecks(unittest.TestCase):
    def invoke(self, namespace, args):
        with redirect_stdout(io.StringIO()):
            namespace['execute'](args)

    def results(self, args):
        return sorted(args.out.glob('cases/*/npj-generic/result.json'))

    def assert_zero_failure(self, path):
        result = json.loads(path.read_text())
        self.assertFalse(result['rgb_resampled'])
        self.assertTrue(result['predictions_sha256'] == digest(path.parent/'query-predictions.npz'))
        for domain in ('grid', 'paper'):
            scored = result['scores']['primary']['domains'][domain]
            self.assertEqual((scored['g'], scored['w'], scored['c']), (0., 0., 0.))
            self.assertIsNone(scored['r'])
        return result

    def test_inference_timeout_keeps_every_case_and_explicit_timeout(self):
        class TimeoutModel:
            def infer(self, image):
                raise TimeoutError('Independent QA injected timeout')
        with tempfile.TemporaryDirectory() as temporary:
            args, rows = fixture(temporary)
            self.invoke(extract(lambda *a, **k: TimeoutModel()), args)
            self.assertEqual(len(self.results(args)), len(rows))
            for path in self.results(args):
                result = self.assert_zero_failure(path)
                self.assertEqual(result['status'], 'timeout')

    def test_model_load_failure_still_materializes_all_scheduled_cases(self):
        def load_failure(*args, **kwargs):
            raise RuntimeError('Independent QA injected load failure')
        with tempfile.TemporaryDirectory() as temporary:
            args, rows = fixture(temporary)
            self.invoke(extract(load_failure), args)
            self.assertEqual(len(self.results(args)), len(rows))
            for path in self.results(args):
                result = self.assert_zero_failure(path)
                self.assertIn('Model-load failure', result['metadata']['execution_error'])

    def test_output_budget_is_enforced_as_operational_failure(self):
        class OversizedModel:
            def infer(self, image):
                return None, np.zeros((5, 5, 3)), {}
        with tempfile.TemporaryDirectory() as temporary:
            args, rows = fixture(temporary, pixel_budget=16)
            self.invoke(extract(lambda *a, **k: OversizedModel()), args)
            self.assertEqual(len(self.results(args)), len(rows))
            for path in self.results(args):
                result = self.assert_zero_failure(path)
                self.assertIn('output pixel budget', result['metadata']['execution_error'])

    def test_resume_rejects_modified_prediction_archive(self):
        def load_failure(*args, **kwargs):
            raise RuntimeError('Independent QA injected load failure')
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = fixture(temporary)
            namespace = extract(load_failure)
            self.invoke(namespace, args)
            path = self.results(args)[0].parent/'query-predictions.npz'
            path.write_bytes(path.read_bytes()+b'injected mutation')
            with self.assertRaisesRegex(ValueError, 'prediction archive changed'):
                self.invoke(namespace, args)

    def test_resume_rejects_changed_execution_lock(self):
        def load_failure(*args, **kwargs):
            raise RuntimeError('Independent QA injected load failure')
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = fixture(temporary)
            namespace = extract(load_failure)
            self.invoke(namespace, args)
            path = args.out/'lock.json'
            lock = json.loads(path.read_text())
            lock['injected_change'] = True
            path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError, 'different execution lock'):
                self.invoke(namespace, args)


if __name__ == '__main__':
    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OrchestrationFailureChecks))
    report = {'status': 'pass' if outcome.wasSuccessful() else 'fail', 'tests': outcome.testsRun,
              'failures': len(outcome.failures), 'errors': len(outcome.errors),
              'source_file': str(SOURCE), 'source_sha256': digest(SOURCE),
              'actual_orchestrator_exercised': True, 'actual_neural_inference_exercised': False,
              'scope': 'Controlled injected inference timeout, model-load failure, pixel-budget failure and resume-corruption tests, two scheduled cases each. No model accuracy claim.'}
    (Path(__file__).parent/'orchestrator-failure-review.json').write_text(json.dumps(report, indent=2)+'\n')
    sys.exit(0 if outcome.wasSuccessful() else 1)
