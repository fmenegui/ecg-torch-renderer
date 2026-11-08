"""Independent rectangle-overlap references, without summed-area queries."""
import unittest
import numpy as np

from axis_aligned_area import AxisAlignedAreaSampler
from streaming_sampler import OutputGrid


def scalar_source_cell_overlap(source, matrix, offset, grid, fill_rgb=(255., 255., 255.)):
    source = source.astype(float)
    channels = source.shape[2]
    fill = np.zeros(4) if channels == 4 else np.asarray(fill_rgb)
    if channels == 4:
        source[..., :3] *= source[..., 3:4]/255.
    result = np.empty((*grid.shape_hw, channels), dtype=float)
    fractions = np.empty(grid.shape_hw, dtype=float)
    scale = np.diag(matrix)
    halfwidth = np.abs(scale)*np.asarray(grid.spacing_mm)/2
    for row in range(grid.shape_hw[0]):
        for col in range(grid.shape_hw[1]):
            centre = (np.array([col, row])*grid.spacing_mm+grid.origin_mm)*scale+offset
            lower, upper = centre-halfwidth, centre+halfwidth
            area = np.prod(upper-lower)
            value, retained = np.zeros(channels), 0.
            for sy in range(source.shape[0]):
                for sx in range(source.shape[1]):
                    dx = max(0., min(upper[0], sx+.5)-max(lower[0], sx-.5))
                    dy = max(0., min(upper[1], sy+.5)-max(lower[1], sy-.5))
                    cell_area = dx*dy
                    retained += cell_area
                    value += cell_area*source[sy, sx]
            value += (area-retained)*fill
            value /= area
            if channels == 4:
                if value[3] > 0:
                    value[:3] *= 255./value[3]
                else:
                    value[:3] = 0
            result[row, col], fractions[row, col] = value, retained/area
    return result, fractions


class AxisAreaTests(unittest.TestCase):
    def test_main_32_to_1_attack_positive_control(self):
        source = np.full((65, 65, 3), 255, np.uint8)
        source[:, 32] = 0
        sampler = AxisAlignedAreaSampler(source, np.diag([32., 32.]), [32., 32.])
        image, mean, fraction = sampler.sample(OutputGrid((1, 1), (0., 0.), (1., 1.)))
        np.testing.assert_allclose(mean, 255*31/32, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(image, np.full((1, 1, 3), 247, np.uint8))
        np.testing.assert_array_equal(fraction, [[1.]])

    def test_random_rgb_rgba_against_independent_scalar_overlap(self):
        rng = np.random.default_rng(31077)
        grid = OutputGrid((5, 7), (-1., -.25), (.7, .9))
        for channels in (3, 4):
            source = rng.integers(0, 256, (8, 9, channels), dtype=np.uint8)
            for scale, offset in (([3.25, .75], [-.25, 1.5]), ([-1.5, 2.], [6., -1.]),
                                  ([.25, -.5], [2.125, 3.25])):
                matrix = np.diag(scale)
                sampler = AxisAlignedAreaSampler(source, matrix, offset, fill_rgb=(7., 8., 9.))
                expected, expected_fraction = scalar_source_cell_overlap(source, matrix, offset, grid,
                                                                         fill_rgb=(7., 8., 9.))
                for budget in (1, 7, 1000):
                    with self.subTest(channels=channels, scale=scale, tile=budget):
                        image, mean, fraction = sampler.sample(grid, max_tile_pixels=budget)
                        np.testing.assert_allclose(mean, expected, rtol=0, atol=2e-10)
                        np.testing.assert_allclose(fraction, expected_fraction, rtol=0, atol=2e-14)
                        # A zero-distance tie can change final uint8 by 1 LSB
                        # under different finite summation orders. Keep mean
                        # tolerance explicit rather than hiding quantisation.
                        self.assertLessEqual(np.max(np.abs(image.astype(float)-np.clip(np.rint(expected), 0, 255))), 1.)

    def test_identity_rgb_exact_all_borders(self):
        source = np.random.default_rng(17).integers(0, 256, (11, 13, 3), dtype=np.uint8)
        sampler = AxisAlignedAreaSampler(source, np.eye(2), [0., 0.])
        image, mean, fraction = sampler.sample(OutputGrid((11, 13), (0., 0.), (1., 1.)), max_tile_pixels=7)
        np.testing.assert_array_equal(image, source)
        np.testing.assert_array_equal(mean, source.astype(float))
        np.testing.assert_array_equal(fraction, np.ones((11, 13)))

    def test_partial_source_area_does_not_renormalise_or_erase(self):
        source = np.zeros((3, 3, 3), np.uint8)
        sampler = AxisAlignedAreaSampler(source, np.diag([2., 2.]), [0., 0.])
        image, mean, fraction = sampler.sample(OutputGrid((1, 1), (0., 0.), (1., 1.)))
        np.testing.assert_allclose(fraction, [[.5625]], rtol=0, atol=0)
        np.testing.assert_allclose(mean, 255*(1-.5625), rtol=0, atol=0)
        np.testing.assert_array_equal(image, np.full((1, 1, 3), 112, np.uint8))

    def test_rgba_hidden_rgb_has_zero_weight(self):
        source = np.zeros((5, 5, 4), np.uint8)
        source[2, 2] = [255, 0, 0, 255]
        variant = source.copy()
        variant[source[..., 3] == 0, :3] = [0, 255, 255]
        grid = OutputGrid((3, 3), (1., 1.), (1., 1.))
        a = AxisAlignedAreaSampler(source, np.diag([2., 2.]), [0., 0.]).sample(grid)
        b = AxisAlignedAreaSampler(variant, np.diag([2., 2.]), [0., 0.]).sample(grid)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_rejects_unsupported_geometry_and_unresolved_area(self):
        source = np.zeros((3, 3, 3), np.uint8)
        for matrix in ([[1, .000000000000001], [0, 1]], [[0, -1], [1, 0]],
                       [[0, 0], [0, 1]], [[np.inf, 0], [0, 1]]):
            with self.assertRaises(ValueError):
                AxisAlignedAreaSampler(source, matrix, [0, 0])
        sampler = AxisAlignedAreaSampler(source, np.eye(2), [1e20, 1e20])
        with self.assertRaises(ValueError):
            sampler.sample(OutputGrid((1, 1), (0., 0.), (1., 1.)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
