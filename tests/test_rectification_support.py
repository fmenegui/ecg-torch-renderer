import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ecg_gpu.digitization.artifacts import verify_assets
from ecg_gpu.digitization.support import cell_support_summary


class GeometryArtifactTests(unittest.TestCase):
    def test_geometry_scope_still_verifies_geometry_but_does_not_require_waveform(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'heng/stage0_model.py'
            path.parent.mkdir()
            path.write_bytes(b'reviewed')
            good = {'path': 'heng/stage0_model.py', 'size': 8,
                    'sha256': hashlib.sha256(b'reviewed').hexdigest()}
            absent = {'path': 'third/stage2.pth', 'size': 9, 'sha256': '0'*64}
            with patch('ecg_gpu.digitization.artifacts.artifact_lock',
                       return_value={'artifacts': [good, absent]}):
                result = verify_assets(root, rectification_only=True)
                self.assertEqual(result['verified_paths'], ['heng/stage0_model.py'])
                with self.assertRaises(FileNotFoundError):
                    verify_assets(root)
                path.write_bytes(b'changed!')
                with self.assertRaises(ValueError):
                    verify_assets(root, rectification_only=True)


class SupportTests(unittest.TestCase):
    def test_equal_cell_counts_can_have_very_different_connectivity(self):
        common = {'ids': [[0, 0], [4, 4]]}
        connected = cell_support_summary(dict(common, cells=[[0, 0], [0, 1], [1, 0], [1, 1]]))
        fragmented = cell_support_summary(dict(common, cells=[[0, 0], [0, 2], [2, 0], [2, 2]]))
        self.assertEqual(connected['cell_coverage_fraction'], .25)
        self.assertEqual(connected['connected_components'], 1)
        self.assertEqual(fragmented['connected_components'], 4)
        self.assertEqual(fragmented['largest_component_fraction'], .25)
        self.assertFalse(connected['full_page_coverage_measured'])

    def test_empty_and_offset_relative_id_extents(self):
        self.assertEqual(cell_support_summary({'ids': [], 'cells': []})['cell_coverage_fraction'], 0.)
        result = cell_support_summary({'ids': [[7, -4], [9, -1]], 'cells': [[7, -4], [7, -3]]})
        self.assertEqual(result['observed_extent_cells'], 6)
        self.assertAlmostEqual(result['cell_coverage_fraction'], 1/3)


if __name__ == '__main__':
    unittest.main()
