import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ecg_gpu.digitization.npj_trace import NativeTraceObserver, physical_clock_proposal
from ecg_gpu.digitization.signals import LEADS


class NativeTraceObservationTests(unittest.TestCase):
    def setUp(self):
        source = os.environ.get('ECG_REPRO_NPJ_UPSTREAM')
        if not source or not (Path(source) / 'src/model/lead_identifier.py').is_file():
            self.skipTest('Pinned original source explicitly required')
        sys.path.insert(0, source)
        from src.model.lead_identifier import LeadIdentifier
        from src.model.signal_extractor import SignalExtractor
        identifier = LeadIdentifier.__new__(LeadIdentifier)
        identifier.required_valid_samples = 3
        identifier.target_num_samples = 5000
        identifier.layouts = {'fixture': {'leads': list(LEADS), 'layout': {'cols': 1}, 'rhythm_leads': []}}
        self.model = SimpleNamespace(identifier=identifier, signal_extractor=SignalExtractor())
        self.match = {'layout': 'fixture', 'cost': 0., 'flip': False}
        t = torch.arange(5000, dtype=torch.float64) / 500
        self.raw = torch.stack([1000 + i * 200 - 100 * torch.exp(-((t - 4.) / .02) ** 2) for i in range(12)])

    def run_native(self, raw):
        cropped = self.model.signal_extractor.preprocess_lines(raw)
        normalized = -self.model.identifier.normalize(cropped, avg_pixel_per_mm=20., mv_per_mm=.1)
        return self.model.identifier._canonicalize_lines(normalized, self.match)

    def test_actual_native_arrays_are_identical_and_all_instance_hooks_restored(self):
        baseline = self.run_native(self.raw)
        before = [dict(owner.__dict__) for owner in [self.model.identifier, self.model.signal_extractor]]
        with NativeTraceObserver(self.model) as observer:
            measured = self.run_native(self.raw)
        np.testing.assert_array_equal(baseline, measured)
        self.assertEqual(observer.report['full_aligned_width'], 5000)
        for owner, state in zip([self.model.identifier, self.model.signal_extractor], before):
            self.assertEqual(set(owner.__dict__), set(state))

    def test_leading_occlusion_is_retained_without_native_time_stretch(self):
        observed = self.raw.clone()
        observed[:, :200] = torch.nan
        with NativeTraceObserver(self.model) as observer:
            old = self.run_native(observed)
        observer.report['pixel_spacing_mm'] = {'x': .05, 'y': .05, 'average_pixel_per_mm': 20.}
        new = physical_clock_proposal(SimpleNamespace(model=self.model), observer, source_id='fixture',
            fs=500, samples=5000, origin_x_px=0., origin_basis='independent analytical raster origin',
            paper_speed_mm_s=25., gain_mm_mv=10.)
        self.assertAlmostEqual(int(old[0].argmax()) / 500, 3.750)
        self.assertEqual(np.nanargmax(new.values_mv[0]) / 500, 4.)
        self.assertEqual(np.isnan(new.values_mv).sum(), 2400)
        self.assertFalse(new.metadata['reference_used'])
        self.assertFalse(new.metadata['time_fitted_to_visible_extent'])

    def test_exception_restores_original_methods(self):
        with self.assertRaisesRegex(RuntimeError, 'controlled'), NativeTraceObserver(self.model):
            raise RuntimeError('controlled')
        self.assertNotIn('normalize', self.model.identifier.__dict__)
        self.assertNotIn('preprocess_lines', self.model.signal_extractor.__dict__)
        self.assertNotIn('_canonicalize_lines', self.model.identifier.__dict__)


if __name__ == '__main__':
    unittest.main()
