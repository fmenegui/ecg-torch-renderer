import os
import unittest
from pathlib import Path

import numpy as np
import torch

from reproductions.kaggle.stage0_diagnose import (
    geometry_control,
    infer_tta_chunks,
    marker_observations,
    native_source,
)


class StageZeroDiagnosisChecks(unittest.TestCase):
    def test_serial_forward_retains_all_four_tta_outputs(self):
        class Model:
            def __call__(self, batch):
                x = batch['image'].float()
                return {'marker': x, 'orientation': x.mean((2, 3))}
        batch = {'image': torch.arange(4 * 3 * 8 * 8).reshape(4, 3, 8, 8)}
        for chunks in [1, 2, 4]:
            out = infer_tta_chunks(Model(), batch, chunks)
            torch.testing.assert_close(out['marker'], batch['image'].float(), atol=0, rtol=0)
            self.assertEqual(out['orientation'].shape, (4, 3))
        with self.assertRaises(ValueError):
            infer_tta_chunks(Model(), batch, 0)

    def native(self):
        root = os.environ.get('ECG_REPRO_KAGGLE_ASSETS')
        if not root or not Path(root).is_dir():
            self.skipTest('Pinned Kaggle assets explicitly required')
        return native_source(root)[0]

    def test_original_postprocessor_passes_noncollinear_positive_control(self):
        result = geometry_control(self.native())
        self.assertEqual(result['matched_points'], 9)
        self.assertLess(result['maximum_reprojection_error_px'], 1.5)

    def test_saved_tta_observations_match_native_postprocessing(self):
        native = self.native()
        marker = torch.zeros((1, 14, 96, 128))
        marker[:, 0] = 1.
        for label, x, y in zip([2, 3, 4, 6, 7, 8, 10, 11, 12], [24, 64, 104] * 3, [16] * 3 + [48] * 3 + [80] * 3):
            marker[:, :, y-2:y+3, x-2:x+3] = 0
            marker[:, label, y-2:y+3, x-2:x+3] = 1
        output = {'marker': torch.cat([marker, marker.flip(2), marker.flip(3), marker.flip((2, 3))]),
                  'orientation': torch.eye(8)[[0, 4, 6, 2]]}
        batch = {'tta': [0, 1, 2, 3], 'sH': 96, 'sW': 128, 'scale': 1.}
        saved = marker_observations(batch, output)
        np.testing.assert_array_equal(saved['marker_average_argmax'], marker.argmax(1)[0].numpy())
        image = np.zeros((96, 128, 3), np.uint8)
        _, expected = native.output_to_predict(image, batch, output)
        actual, rotation = native.marker_to_keypoint(image, torch.from_numpy(saved['orientation_native_average']), marker, 1.)
        self.assertEqual(expected, actual)
        self.assertEqual(rotation, 0)


if __name__ == '__main__':
    unittest.main()
