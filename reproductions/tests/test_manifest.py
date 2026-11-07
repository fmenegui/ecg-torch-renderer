import copy
import tempfile
from pathlib import Path
import unittest

from reproductions.common import sha256, write_json
from reproductions.manifest import inference_case, validate_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "image.bin").write_bytes(b"image")
        (self.root / "ref.bin").write_bytes(b"reference")
        self.case = {"case_id": "x", "patient_id": "source-p1", "split": "test", "image": "image.bin", "image_sha256": sha256(self.root / "image.bin"), "reference": "ref.bin", "reference_sha256": sha256(self.root / "ref.bin"), "reference_unit": "mV", "fs": 500, "acquisition": "scan", "gold_layout": "6x2"}
        self.manifest = {"schema": "ecg-reproduction-manifest-v1", "cases": [self.case]}

    def test_valid_bytes(self):
        self.assertEqual(len(validate_manifest(self.manifest, self.root)), 1)

    def test_patient_leakage_rejected(self):
        other = {**self.case, "case_id": "second-view", "split": "train"}
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            validate_manifest({**self.manifest, "cases": [self.case, other]}, self.root)

    def test_mutated_input_rejected(self):
        (self.root / "image.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Changed image"):
            validate_manifest(self.manifest, self.root)

    def test_gold_is_excluded_from_inference(self):
        view = inference_case(self.case)
        self.assertEqual(set(view), {"case_id", "image", "image_sha256"})

    def test_unknown_metadata_cannot_leak(self):
        with self.assertRaises(ValueError):
            inference_case({**self.case, "public_metadata": {"gold_layout": "6x2"}})

    def test_unknown_units_rejected(self):
        case = copy.deepcopy(self.case)
        case["reference_unit"] = "unknown"
        with self.assertRaises(ValueError):
            validate_manifest({**self.manifest, "cases": [case]}, self.root)

    def test_json_perfect_status_remains_interoperable(self):
        write_json(self.root / "receipt.json", {"snr": float("inf"), "status": "perfect"})
        self.assertNotIn("Infinity", (self.root / "receipt.json").read_text())


if __name__ == "__main__":
    unittest.main()
