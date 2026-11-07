"""A failed image write must never replace an existing valid output."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from examples.rectify_denoising_grid import _write_rgb_png


class PNGIntegrityTests(unittest.TestCase):
    def test_pixel_exact_png_is_published(self):
        rgb = np.random.default_rng(831).integers(0, 256, (31, 47, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "ecg.png"
            _write_rgb_png(rgb, filename)
            with Image.open(filename) as decoded:
                np.testing.assert_array_equal(np.asarray(decoded), rgb)
            self.assertEqual(list(Path(temporary).iterdir()), [filename])

    def test_truncated_write_preserves_previous_complete_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "ecg.png"
            original = np.full((16, 24, 3), 240, np.uint8)
            _write_rgb_png(original, filename)
            before = filename.read_bytes()

            def broken_save(image, path, **kwargs):
                Path(path).write_bytes(b"partial PNG")

            with patch.object(Image.Image, "save", broken_save):
                with self.assertRaises(OSError):
                    _write_rgb_png(np.zeros_like(original), filename)
            self.assertEqual(filename.read_bytes(), before)

    def test_changed_pixels_in_encoded_image_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "ecg.png"
            original = np.full((16, 24, 3), 240, np.uint8)
            _write_rgb_png(original, filename)
            before = filename.read_bytes()
            save = Image.Image.save

            def changed_save(image, path, **kwargs):
                changed = np.asarray(image).copy()
                changed[8, 12] = [255, 0, 0]
                save(Image.fromarray(changed), path, **kwargs)

            with patch.object(Image.Image, "save", changed_save):
                with self.assertRaisesRegex(OSError, "RGB pixels"):
                    _write_rgb_png(np.zeros_like(original), filename)
            self.assertEqual(filename.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
