import gzip
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch

from ecg_gpu.gpu_module import ECGImageLayer
from ecg_gpu.gpu_profiling import profile_layer, region_annotations, _trace_kernel_rows
from ecg_gpu.gpu_renderer import RenderConfig


class DetailedProfilingTests(unittest.TestCase):
    def test_kernel_counts_exclude_annotations_and_memory_transfers(self):
        events = [
            {'name': 'triton_coverage', 'ph': 'X', 'cat': 'kernel', 'dur': 40.},
            {'name': 'triton_coverage', 'ph': 'X', 'cat': 'kernel', 'dur': 60.},
            {'name': 'ecg/deform', 'ph': 'X', 'cat': 'gpu_user_annotation', 'dur': 900.},
            {'name': 'cudaLaunchKernel', 'ph': 'X', 'cat': 'cuda_runtime', 'dur': 300.},
            {'name': 'Memcpy DtoH', 'ph': 'X', 'cat': 'gpu_memcpy', 'dur': 100.},
            {'name': 'Memset', 'ph': 'X', 'cat': 'gpu_memset', 'dur': 50.},
            {'name': 'triton_coverage', 'ph': 'M', 'cat': 'kernel', 'dur': 20.},
        ]
        rows = _trace_kernel_rows(events)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['kernel'], 'triton_coverage')
        self.assertEqual(rows[0]['calls'], 2)
        self.assertAlmostEqual(rows[0]['total_ms'], .1)
        self.assertEqual(_trace_kernel_rows([]), [])

    def setUp(self):
        torch.set_num_threads(2)
        self.layer = ECGImageLayer(fs=100, seed=19, config=RenderConfig(
            height=64, width=96, antialias_samples=1, surface_resolution=32,
            inverse_iterations=2, shadow_samples=0))
        self.signals = torch.zeros(1, 12, 1000)

    def test_trace_archive_and_cpu_metrics_are_consistent(self):
        original = self.layer.renderer._curves
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'profile'
            result = profile_layer(self.layer, self.signals, destination,
                                   steps=2, warmup=1, profile_steps=1)
            self.assertFalse(result['cuda_measured'])
            self.assertEqual(result['profile']['cuda_kernel_events'], 0)
            self.assertEqual(result['profile']['top_kernels'], [])
            self.assertIsNone(result['timing']['cuda_event_ms_median'])
            self.assertIsNone(result['timing']['cuda_peak_allocated_bytes'])
            self.assertGreater(result['timing']['median_ms'], 0)
            self.assertEqual(self.layer._calls, 5)
            self.assertEqual(self.layer.renderer._curves, original)
            with gzip.open(destination / 'trace.json.gz', 'rt') as stream:
                trace = json.load(stream)
            names = {event.get('name') for event in trace['traceEvents']}
            self.assertIn('ecg/_curves', names)
            self.assertIn('ecg/step_0', names)
            with zipfile.ZipFile(result['archive']) as archive:
                self.assertIsNone(archive.testzip())
                stored = json.loads(archive.read('report.json'))
                self.assertEqual(stored['timing'], result['timing'])
                self.assertFalse(stored['signals_or_patient_metadata_exported'])
            with self.assertRaisesRegex(ValueError, 'must be empty'):
                profile_layer(self.layer, self.signals, destination)

    def test_region_wrappers_restore_after_failure(self):
        original = self.layer.renderer._curves
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with region_annotations(self.layer):
                raise RuntimeError('interrupted')
        self.assertEqual(self.layer.renderer._curves, original)


if __name__ == '__main__':
    unittest.main()
