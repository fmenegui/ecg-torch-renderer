"""Physical metadata and stage-isolation controls for external geometry."""

import threading
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ecg_gpu.digitization.rectified_backend import RectifiedNpjBackend
from ecg_gpu.digitization.oriented_rectified_backend import LandscapeRectifiedNpjBackend


class RectifiedBackendTests(unittest.TestCase):
    def model(self):
        received = {}

        def normalize(image):
            received["input_shape"] = tuple(image.shape)
            return image / 255

        def features(image):
            received["pixels"] = image.clone()
            return image[:, :1], image[:, 1:2], image[:, 2:3]

        def identify(signals, text, density, *, layout_should_include_substring):
            received["signal_shape"] = tuple(signals.shape)
            received["signal_pixels"] = signals.clone()
            received["density"] = density
            received["hint"] = layout_should_include_substring
            # A 20-pixel deflection at 4 px/mm, printed at 10 mm/mV,
            # is 0.5 mV. The original native module returns microvolts.
            values = torch.full((12, 1000), 20 / density / 10 * 1000)
            values[:, 5:15] = torch.nan
            return {"layout": "standard_12x1", "cost": .4, "canonical_lines": values}

        def prohibited(*args):
            raise AssertionError("A second geometry or pitch inference is prohibited")

        model = SimpleNamespace(min_max_normalize=normalize, _get_feature_maps=features,
            signal_extractor=lambda image: image, identifier=identify,
            perspective_detector=prohibited, cropper=prohibited, dewarper=prohibited,
            pixel_size_finder=prohibited, _resample_image=prohibited, _rotate_on_resample=prohibited)
        native = SimpleNamespace(model=model, max_side=256, device="cpu", _lock=threading.Lock())
        return native, received

    def test_explicit_density_preserves_amplitude_axes_and_missingness_without_second_geometry(self):
        native, received = self.model()
        rgb = np.zeros((80, 128, 3), np.uint8)
        rgb[3, 17] = [255, 34, 79]
        result = RectifiedNpjBackend(native).predict(rgb, fs=500, samples=1000, source_id="one-source", pixels_per_mm=4)
        self.assertEqual(result.fs, 500)
        self.assertEqual(result.values_mv.shape, (12, 1000))
        self.assertEqual(received["input_shape"], (1, 3, 80, 128))
        np.testing.assert_array_equal(received["pixels"][0, :, 3, 17].numpy(), rgb[3, 17] / np.float32(255))
        self.assertEqual(received["density"], 4)
        self.assertIsNone(received["hint"])
        np.testing.assert_array_equal(result.values_mv[:, :5], np.full((12, 5), .5))
        self.assertTrue(np.isnan(result.values_mv[:, 5:15]).all())
        self.assertFalse(result.metadata["native_perspective_applied"])
        self.assertFalse(result.metadata["original_article_full_pipeline"])

    def test_invalid_external_density_and_constant_or_oversized_canvas_are_rejected(self):
        native, _ = self.model()
        adapter = RectifiedNpjBackend(native)
        image = np.arange(32 * 40 * 3, dtype=np.uint8).reshape(32, 40, 3)
        for density in [0, -1, True, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                adapter.predict(image, fs=500, samples=1000, source_id="one", pixels_per_mm=density)
        for bad in [np.zeros_like(image), np.zeros((400, 32, 3), np.uint8)]:
            with self.assertRaises(ValueError):
                adapter.predict(bad, fs=500, samples=1000, source_id="one", pixels_per_mm=4)

    def test_landscape_control_turns_maps_exactly_without_patching_the_shared_native_model(self):
        native, received = self.model()
        original_method = native.model._get_feature_maps
        image = np.zeros((128, 80, 3), np.uint8)
        image[3, 17, 0] = 255
        result = LandscapeRectifiedNpjBackend(native).predict(image, fs=500, samples=1000,
            source_id="portrait", pixels_per_mm=4)
        self.assertIs(native.model._get_feature_maps, original_method)
        self.assertEqual(received["signal_shape"], (80, 128))
        # (x=17,y=3) becomes (x=124,y=17) after one clockwise turn.
        self.assertEqual(float(received["signal_pixels"][17, 124]), 1)
        self.assertEqual(float(received["signal_pixels"].sum()), 1)
        self.assertEqual(result.metadata["feature_map_quarter_turns_clockwise"], 1)
        self.assertFalse(result.metadata["semantic_orientation_verified"])
        self.assertEqual(received["density"], 4)


if __name__ == "__main__":
    unittest.main()
