from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.prepare_pm_ecg_external import fetch_member, make_reference


class PMECGReferenceTest(unittest.TestCase):
    def test_short_windows_keep_standard_column_ownership(self):
        short = np.arange(1250 * 12).reshape(1250, 12).astype(float)
        reference, names = make_reference(short, np.empty((5000, 0)), 4)
        self.assertEqual(names, [])
        self.assertEqual(np.isfinite(reference).sum(), 15000)
        for lead, col in enumerate([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3):
            np.testing.assert_array_equal(
                reference[lead, col * 1250 : (col + 1) * 1250], short[:, lead]
            )
            self.assertEqual(np.isfinite(reference[lead]).sum(), 1250)

    def test_rhythm_identity_requires_exact_reference_evidence(self):
        short = np.arange(1250 * 12).reshape(1250, 12).astype(float)
        rhythm = np.full((5000, 1), -5.0)
        rhythm[:1250, 0] = short[:, 1]
        reference, names = make_reference(short, rhythm, 4)
        self.assertEqual(names, ["II"])
        np.testing.assert_array_equal(reference[1], rhythm[:, 0])
        self.assertEqual(np.isfinite(reference).sum(), 18750)

    def test_ambiguous_rhythm_is_rejected(self):
        with self.assertRaises(ValueError):
            make_reference(np.zeros((1250, 12)), np.zeros((5000, 1)), 4)

    def test_cached_member_crc_is_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "signal.bin").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                fetch_member(root, {"name": "signal.bin", "size": 3, "crc": 0})


if __name__ == "__main__":
    unittest.main()
