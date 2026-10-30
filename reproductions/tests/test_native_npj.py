import copy
import os
from pathlib import Path
import sys
import unittest

import numpy as np

from reproductions.npj.protocol import COMMIT, inference_config, training_config
from reproductions.npj.train import optimizer_state, restore_optimizer

UPSTREAM = os.getenv("ECG_REPRO_NPJ_UPSTREAM")


@unittest.skipUnless(UPSTREAM, "Set ECG_REPRO_NPJ_UPSTREAM to the pinned original source")
class NativeNpjTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from reproductions.common import verify_source
        verify_source(UPSTREAM, COMMIT)
        sys.path.insert(0, str(Path(UPSTREAM).resolve()))

    def test_table4_and_public_config_are_distinct(self):
        paper = training_config(UPSTREAM, "paper_table4", "/train", "/val")
        public = training_config(UPSTREAM, "public_config", "/train", "/val")
        self.assertEqual(paper["TRAIN"]["KWARGS"]["epochs"], 45)
        self.assertEqual(public["TRAIN"]["KWARGS"]["epochs"], 100)
        self.assertEqual(paper["TRAIN"]["OPTIMIZER"]["KWARGS"]["weight_decay"], 0.001)
        self.assertEqual(paper["TRAIN"]["LR_SCHEDULER"]["KWARGS"]["T_max"], 20000)

    def test_original_perspective_presets_disable_experimental_tps(self):
        for preset in ["ahus", "george-moody-2024", "generic"]:
            cfg = inference_config(UPSTREAM, preset, "cpu")
            self.assertFalse(cfg["MODEL"]["KWARGS"]["apply_dewarping"])
            self.assertEqual(cfg["MODEL"]["KWARGS"]["resample_size"], 3000)

    def test_public_sampling_override_does_not_change_native_default(self):
        default = inference_config(UPSTREAM, "george-moody-2024", "cpu")
        adapted = inference_config(UPSTREAM, "george-moody-2024", "cpu", 5000)
        self.assertEqual(default["MODEL"]["KWARGS"]["config"]["LAYOUT_IDENTIFIER"]["KWARGS"]["target_num_samples"], 10000)
        self.assertEqual(adapted["MODEL"]["KWARGS"]["config"]["LAYOUT_IDENTIFIER"]["KWARGS"]["target_num_samples"], 5000)
        with self.assertRaises(ValueError):
            inference_config(UPSTREAM, "george-moody-2024", "cpu", 0)

    def test_inner_optimizer_resume_matches_uninterrupted(self):
        import torch
        from src.optimizer.adammuon import AdamMuon
        torch.manual_seed(2)
        a = torch.nn.Linear(3, 2)
        b = copy.deepcopy(a)
        oa, ob = AdamMuon(a.parameters(), lr=0.0037), AdamMuon(b.parameters(), lr=0.0037)
        x = torch.tensor([[0.2, 0.7, -1.0], [1.1, 0.1, 0.8]])

        def step(model, optimizer):
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()

        step(a, oa)
        b.load_state_dict(a.state_dict())
        restore_optimizer(ob, copy.deepcopy(optimizer_state(oa)))
        step(a, oa)
        step(b, ob)
        for pa, pb in zip(a.parameters(), b.parameters()):
            np.testing.assert_array_equal(pa.detach().numpy(), pb.detach().numpy())

    def test_native_evaluator_control_is_preserved(self):
        from src.evaluate import compute_metrics
        y = np.sin(np.arange(100) * 0.127)[:, None]
        native = compute_metrics(y.copy(), y.copy(), n_translations=0)
        self.assertTrue(np.isnan(native["snr_db"][0]))

    def test_native_extent_normalization_time_control_is_not_silently_replaced(self):
        from reproductions.npj.time_axis_control import measure
        result = measure(UPSTREAM)
        self.assertEqual(result["rows"][0]["native_peak_s"], 4.)
        hidden = result["rows"][2]
        self.assertEqual(hidden["physical_missing_duration_s"], .4)
        self.assertEqual(hidden["input_missing_samples"], 600)
        self.assertEqual(hidden["native_output_missing_samples"], 0)
        self.assertGreater(abs(hidden["native_peak_error_ms"]), 200)
        self.assertFalse(result["real_photo_error_cause_proven"])


if __name__ == "__main__":
    unittest.main()
