import hashlib
from pathlib import Path
import tempfile
import unittest

from reproductions.npj.data_identity import source_identity, validate_receipt_source, verify_shards


class DataIdentityTests(unittest.TestCase):
    def setUp(self):
        self.lock = {"repository": "author/dataset", "revision": "pinned", "files": [
            {"path": "data/train-0.parquet", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()},
            {"path": "data/val-0.parquet", "bytes": 3, "sha256": hashlib.sha256(b"xyz").hexdigest()},
        ]}

    def test_complete_byte_verified_release_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train-0.parquet").write_bytes(b"abc")
            (root / "val-0.parquet").write_bytes(b"xyz")
            shards, complete = verify_shards(root, self.lock, require_complete=True)
            self.assertTrue(complete)
            receipt = {"source_release": source_identity(self.lock), "shards": shards}
            validate_receipt_source(receipt, self.lock)
            receipt["shards"][0]["sha256"] = "changed"
            with self.assertRaises(ValueError):
                validate_receipt_source(receipt, self.lock)

    def test_matching_length_and_filename_do_not_accept_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train-0.parquet").write_bytes(b"abd")
            with self.assertRaises(ValueError):
                verify_shards(root, self.lock)

    def test_partial_and_duplicate_shards_cannot_be_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train-0.parquet").write_bytes(b"abc")
            self.assertFalse(verify_shards(root, self.lock)[1])
            with self.assertRaises(ValueError):
                verify_shards(root, self.lock, require_complete=True)
            (root / "duplicate").mkdir()
            (root / "duplicate/train-0.parquet").write_bytes(b"abc")
            with self.assertRaises(ValueError):
                verify_shards(root, self.lock)


if __name__ == "__main__":
    unittest.main()
