"""Independent controls of the actual M3 measurement orchestration.

AST extraction prevents checkpoint loading. The map class and detector method
are the exact source under review, not a second implementation of them.
"""
import ast
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
SOURCE = REPO/'examples/compare_rectification_methods.py'
sys.path.insert(0, str(REPO))
from ecg_gpu.digitization.cell_mesh import CellMap, cell_triangles
from ecg_gpu.digitization.complete_rectifier import CompleteMap


def extracted():
    names = {'OurNativeMap', 'OurDetector', 'deadline'}
    nodes = [n for n in ast.parse(SOURCE.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    namespace = {'np': np, 'time': time, 'CellMap': CellMap,
                 'cell_triangles': cell_triangles, 'CompleteMap': CompleteMap,
                 'contextmanager': contextmanager, 'signal': signal}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


def archive(folder):
    observed = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
    canonical = observed/2
    identities = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    np.savez_compressed(folder/'observed-grid-geometry.npz', xy=observed,
                        canonical_mm=canonical, ids=identities, cells=[[0, 0]])
    source = np.array([[0., 0.], [20., 0.], [20., 20.], [0., 20.]])
    np.savez_compressed(folder/'full-frame-mesh.npz', source=source,
                        target_mm=source/2, triangles=[[0, 1, 2], [0, 2, 3]])


class OrchestrationGeometryChecks(unittest.TestCase):
    def test_original_preserved_keeps_partial_coordinates_without_rgb_footprint(self):
        namespace = extracted()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            archive(folder)
            native = namespace['OurNativeMap'](folder, {'complete_geometry': {'resampled': False}})
            coordinates, eligible, footprint = native.predict(np.array([[5., 5.], [15., 15.]]))
            np.testing.assert_allclose(coordinates[0], [2.5, 2.5], atol=1e-12)
            self.assertTrue(np.isnan(coordinates[1]).all())
            np.testing.assert_array_equal(eligible, [True, False])
            self.assertFalse(footprint.any())

    def test_actual_resampling_archived_full_map_is_separate_from_confidence(self):
        namespace = extracted()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            archive(folder)
            native = namespace['OurNativeMap'](folder, {'complete_geometry': {'resampled': True}})
            coordinates, eligible, footprint = native.predict(np.array([[5., 5.], [15., 15.], [25., 25.]]))
            np.testing.assert_allclose(coordinates[:2], [[2.5, 2.5], [7.5, 7.5]], atol=1e-12)
            np.testing.assert_array_equal(eligible, [True, False, False])
            np.testing.assert_array_equal(footprint, [True, True, False])

    def detector_control(self, apply_banding):
        namespace = extracted()
        detector = namespace['OurDetector'].__new__(namespace['OurDetector'])
        detector.model = object()
        detector.threshold = .85
        image = np.zeros((8, 8, 3), np.uint8)
        filtered = np.ones_like(image)
        calls = []
        pilot = np.array([[1., 1.], [2., 2.]])
        final = pilot+[.125, .25]

        def detect(model, rgb, *, device, threshold):
            self.assertIs(model, detector.model)
            self.assertEqual(device, 'cpu')
            self.assertEqual(threshold, .85)
            calls.append(('detect', rgb is image))
            return (pilot if rgb is image else final).copy(), np.array([.9, .95]), object()

        def banding(rgb, pilot_points):
            self.assertIs(rgb, image)
            np.testing.assert_array_equal(pilot_points, pilot)
            calls.append(('banding', True))
            return filtered, {'applied': apply_banding}

        namespace.update(detect=detect, remove_periodic_banding=banding,
                         torch=SimpleNamespace(inference_mode=nullcontext),
                         organize_lattice=lambda xy, scores: {},
                         locally_supported_cells=lambda xy, topology: {},
                         CHECKPOINT_SHA='analytic-control')
        with tempfile.TemporaryDirectory() as temporary:
            result = detector.infer(image, Path(temporary), Path(temporary)/'absent.npz')
        expected = [('detect', True), ('banding', True)]
        if apply_banding:
            expected.append(('detect', False))
        self.assertEqual(calls, expected)
        np.testing.assert_array_equal(result[0], final if apply_banding else pilot)

    def test_detector_original_then_banding_then_conditional_redetection(self):
        self.detector_control(True)

    def test_detector_does_not_redetect_when_banding_is_not_applied(self):
        self.detector_control(False)

    def test_cooperative_deadline_records_timeout_and_restores_handler(self):
        deadline = extracted()['deadline']
        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with deadline(.01):
                time.sleep(.05)
        self.assertIs(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OrchestrationGeometryChecks)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'schema': 'M3-independent-orchestration-geometry-QA-v1',
              'source_file': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'tests': outcome.testsRun, 'failures': len(outcome.failures), 'errors': len(outcome.errors),
              'actual_wrapper_source_exercised': True, 'actual_neural_inference_exercised': False,
              'deadline_scope': 'Cooperative SIGALRM at Python interruption points. This test does not prove hard preemption of long C/C++ inference operations.',
              'status': 'pass' if outcome.wasSuccessful() else 'fail'}
    (Path(__file__).parent/'orchestrator-geometry-review.json').write_text(json.dumps(report, indent=2)+'\n')
    sys.exit(0 if outcome.wasSuccessful() else 1)
