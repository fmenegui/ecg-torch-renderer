"""Sampling checks against independent analytical coordinate/color fields."""

import unittest

import numpy as np

from ecg_gpu.digitization.adaptive_sampling import choose_density, render_mesh, validate_mesh


class AdaptiveSamplingTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array([[0, 0], [99, 0], [99, 79], [0, 79]], float)
        self.target = self.source / 8
        self.triangles = np.array([[0, 1, 2], [0, 2, 3]])
        y, x = np.mgrid[:80, :100]
        self.rgb = np.stack((x, y, x + y), -1).astype(np.uint8)

    def test_identity_sampling_preserves_every_original_rgb_pixel(self):
        image, valid, report = render_mesh(self.rgb, self.source, self.target, self.triangles, max_side=200)
        np.testing.assert_array_equal(image[1:81, 1:101], self.rgb)
        self.assertTrue(valid[1:81, 1:101].all())
        self.assertEqual(report["effective_pixels_per_mm"], 8)
        self.assertFalse(report["budget_requires_downsampling"])

    def test_density_tracks_independent_mm_to_pixel_scale(self):
        for scale in [6, 12, 20]:
            density, report = choose_density(self.source, self.source / scale, self.triangles, max_side=200)
            self.assertAlmostEqual(density, scale)
            self.assertFalse(report["budget_requires_downsampling"])

    def test_canvas_budget_is_explicit_and_never_silently_overrun(self):
        image, _, report = render_mesh(self.rgb, self.source, self.target, self.triangles, max_side=50)
        self.assertLessEqual(max(image.shape[:2]), 50)
        self.assertTrue(report["budget_requires_downsampling"])
        with self.assertRaises(ValueError):
            render_mesh(self.rgb, self.source, self.target, self.triangles, max_side=50, pixels_per_mm=8)

    def test_crop_missing_triangles_and_reflections_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_mesh(self.source, self.target, self.triangles[:1], self.rgb.shape)
        reflected = self.target.copy()
        reflected[:, 0] *= -1
        with self.assertRaises(ValueError):
            validate_mesh(self.source, reflected, self.triangles, self.rgb.shape)

    def test_global_disconnection_and_overlap_are_rejected(self):
        source = self.source[self.triangles].reshape(-1, 2)
        target = self.target[self.triangles].reshape(-1, 2)
        triangles = np.array([[0, 1, 2], [3, 4, 5]])
        target[3:] += [20, 0]
        with self.assertRaises(ValueError):
            validate_mesh(source, target, triangles, self.rgb.shape)
        target[3:] -= [25, 0]
        with self.assertRaises(ValueError):
            validate_mesh(source, target, triangles, self.rgb.shape)

    def test_given_diagonal_is_retained_for_a_non_affine_map(self):
        target = self.target.copy()
        target[2] += [1, 0.5]
        out, valid, report = render_mesh(self.rgb, self.source, target, self.triangles, max_side=200, pixels_per_mm=8)
        # A query strictly inside triangle 0. Barycentric interpolation provides
        # an independent physical reference, without calling the implementation.
        q = np.mean(target[self.triangles[0]], axis=0)
        pixel = np.rint(q * 8 - np.asarray(report["output_origin_mm"]) * 8).astype(int)
        physical = (pixel + np.asarray(report["output_origin_mm"]) * 8) / 8
        vertices = target[self.triangles[0]]
        weights = np.linalg.solve(np.vstack((vertices.T, np.ones(3))), np.r_[physical, 1])
        xy = weights @ self.source[self.triangles[0]]
        expected = np.array([xy[0], xy[1], xy.sum()])
        self.assertTrue(valid[pixel[1], pixel[0]])
        np.testing.assert_allclose(out[pixel[1], pixel[0]], expected, atol=0.55)


if __name__ == "__main__":
    unittest.main()
