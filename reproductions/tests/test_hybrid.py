"""Independent support, reference isolation and owned-infrastructure controls."""

import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from PIL import Image

from ecg_gpu.digitization.selective import SelectionPolicy, SignalProposal
from reproductions.hybrid.recover_checkpoint import Recovery, VOLUME, choose_recovery
from reproductions.hybrid.worker import infer_images


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.gpus = [{"gpuId": "NVIDIA A40", "securePricePerHr": .59, "memoryInGb": 48,
                      "dataCenterAvailability": [{"dataCenterId": "CA-MTL-1", "stockStatus": "Medium"}]}]
        self.templates = [{"id": "runpod-torch-v280", "imageName": "runpod/pytorch:catalog"}]
        self.volume = {"id": VOLUME, "name": "ecg-diverse-offline-bank", "dataCenter": "CA-MTL-1", "size": 50}

    def test_recovery_requires_observed_price_stock_and_exact_borrowed_volume(self):
        self.assertEqual(choose_recovery(self.gpus, self.templates, self.volume)["gpu_cost_cap_usd"], .15)
        cheaper = dict(self.gpus[0], gpuId="NVIDIA L4", securePricePerHr=.39, memoryInGb=24)
        self.assertEqual(choose_recovery([*self.gpus, cheaper], self.templates, self.volume)["gpu"]["gpuId"], "NVIDIA L4")
        for volume in [dict(self.volume, size=60), dict(self.volume, id="someone-else"), dict(self.volume, dataCenter="elsewhere")]:
            with self.assertRaises(ValueError):
                choose_recovery(self.gpus, self.templates, volume)
        for price in [.61, None, 0]:
            with self.assertRaises(ValueError):
                choose_recovery([dict(self.gpus[0], securePricePerHr=price)], self.templates, self.volume)
        with self.assertRaises(ValueError):
            choose_recovery([dict(self.gpus[0], dataCenterAvailability=[])], self.templates, self.volume)

    def test_recovery_cannot_delete_or_own_a_borrowed_volume(self):
        controller = Recovery.__new__(Recovery)
        controller.ledger = {"borrowed_volume_id": VOLUME}
        with self.assertRaises(ValueError):
            controller.cleanup(delete_volume=True)
        controller.ledger["volume_id"] = VOLUME
        with self.assertRaises(ValueError):
            controller.cleanup()


class BlindedPipelineTests(unittest.TestCase):
    def test_reference_fields_cannot_enter_the_inference_boundary(self):
        for extra in ["reference", "true_layout", "printed_mask", "reference_waveforms"]:
            with self.assertRaises(ValueError):
                infer_images([{"image_id": "one", "image": "one.png", "image_sha256": "abc", extra: "forbidden"}],
                             None, None, None, None, None, SelectionPolicy(), time.time() + 500)

    def test_failed_candidates_keep_exact_direct_and_every_scheduled_result(self):
        t = np.arange(5000) / 500
        values = np.array([np.sin((1 + k / 3) * t) for k in range(12)])
        values[:, 123:175] = np.nan

        class Backend:
            def predict(self, image, *, fs, samples, source_id):
                return SignalProposal(values, fs, source_id, "12x1", metadata={"native_layout": "standard_12x1"})

        class FailedRectifier:
            last_report = None

            def __call__(self, image, *, source_id):
                raise ValueError("Independent deliberate candidate failure")

        def no_specialist(*args):
            raise AssertionError("Ineligible layout must not invoke the specialist")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "one.png"
            Image.fromarray(np.full((32, 48, 3), 129, np.uint8)).save(image)
            cases = [{"image_id": "one", "image": "one.png", "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]
            report = infer_images(cases, root, root / "run", Backend(), FailedRectifier(), no_specialist,
                                  SelectionPolicy(), time.time() + 600)
            self.assertEqual(len(report["cases"]), 4)
            self.assertEqual({r["arm"] for r in report["cases"]}, {"npj_native", "grid_adaptive", "kaggle_native_eligible", "selective"})
            with np.load(root / "run/one/selective.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z["values_mv"], values)
                self.assertEqual(z["fs"], 500)
            audit = json.loads((root / "run/phase1-decisions.json").read_text())
            self.assertFalse(audit["inference_received_reference_paths"])
            self.assertFalse(audit["inference_received_true_layout"])
            self.assertEqual(audit["decisions"][0]["selected"], "direct")


if __name__ == "__main__":
    unittest.main()
