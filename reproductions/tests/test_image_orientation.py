import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from ecg_gpu.digitization.image_io import decode_ecg_rgb, exif_pixel_map
from ecg_gpu.digitization.npj_backend import NativeNpjBackend
from ecg_gpu.digitization.selective import SignalProposal


class ImageOrientationChecks(unittest.TestCase):
    def test_all_eight_exif_maps_match_independent_pixel_permutations(self):
        raw = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        with tempfile.TemporaryDirectory() as folder:
            for orientation in range(1, 9):
                image = Image.fromarray(raw)
                exif = image.getexif()
                exif[274] = orientation
                path = Path(folder) / f'{orientation}.png'
                image.save(path, exif=exif)
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                decoded, receipt = decode_ecg_rgb(path)
                matrix = np.array(receipt['stored_to_decoded_pixel_centers'])
                for y in range(3):
                    for x in range(5):
                        xx, yy, _ = (matrix @ [x, y, 1]).astype(int)
                        np.testing.assert_array_equal(decoded[yy, xx], raw[y, x])
                np.testing.assert_allclose(matrix @ np.array(receipt['decoded_to_stored_pixel_centers']), np.eye(3), atol=0, rtol=0)
                self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
                canonical = Path(folder) / f'canonical-{orientation}.png'
                Image.fromarray(decoded).save(canonical)
                repeated, metadata = decode_ecg_rgb(canonical)
                np.testing.assert_array_equal(repeated, decoded)
                self.assertFalse(metadata['orientation_applied'])

    def test_invalid_orientation_cannot_silently_be_guessed(self):
        for code in [0, 9, True, 6.0, None]:
            with self.assertRaises(ValueError):
                exif_pixel_map(code, 5, 3)

    def test_native_file_entrypoint_preserves_source_identity_and_explicit_axes(self):
        raw = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        with tempfile.TemporaryDirectory() as folder:
            image = Image.fromarray(raw)
            exif = image.getexif()
            exif[274] = 8
            path = Path(folder) / 'phone.png'
            image.save(path, exif=exif)
            rgb, decoding = decode_ecg_rgb(path)
            proposal = SignalProposal(np.full((12, 5), np.nan), 500., decoding['source_id'], None)
            backend = NativeNpjBackend.__new__(NativeNpjBackend)
            backend.predict = Mock(return_value=proposal)
            returned = backend.predict_file(path, fs=500., samples=5)
            args, kwargs = backend.predict.call_args
            np.testing.assert_array_equal(args[0], rgb)
            self.assertEqual(kwargs, {'fs': 500., 'samples': 5, 'source_id': decoding['source_id']})
            self.assertIs(returned, proposal)
            self.assertEqual(returned.metadata['input_rgb_decoding']['exif_orientation'], 8)


if __name__ == '__main__':
    unittest.main()
