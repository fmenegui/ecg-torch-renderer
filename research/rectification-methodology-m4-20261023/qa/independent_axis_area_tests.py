"""Failure-directed review of the separate diagonal-affine area control.

The32:1 defect was known before this control existed. These are development
checks against independent scalar rectangle clipping, not a fresh assessment.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/"m4/sampling"))
from axis_aligned_area import AxisAlignedAreaSampler
from streaming_sampler import OutputGrid

NOTES = {}


def rectangle_reference(source_u8, matrix, offset, grid, fill):
    """Clip every source cell to each output footprint. No prefix sums."""
    source = source_u8.astype(float)
    h, w, channels = source.shape
    fill = np.array(fill, float)
    if channels == 4:
        source[..., :3] *= source[..., 3:4]/255.
        fill = np.zeros(4)
    mean = np.empty((*grid.shape_hw, channels), float)
    fractions = np.empty(grid.shape_hw, float)
    for row in range(grid.shape_hw[0]):
        for col in range(grid.shape_hw[1]):
            physical = np.array([grid.origin_mm[0]+col*grid.spacing_mm[0],
                                 grid.origin_mm[1]+row*grid.spacing_mm[1]])
            centre = matrix@physical+offset
            half = np.abs(np.diag(matrix))*grid.spacing_mm/2
            left, top = centre-half
            right, bottom = centre+half
            area = (right-left)*(bottom-top)
            total = fill*area
            inside_area = 0.
            for y in range(h):
                overlap_y = max(0., min(bottom, y+.5)-max(top, y-.5))
                for x in range(w):
                    overlap_x = max(0., min(right, x+.5)-max(left, x-.5))
                    weight = overlap_x*overlap_y
                    total += (source[y, x]-fill)*weight
                    inside_area += weight
            average = total/area
            if channels == 4:
                # Direct clipping has exact zero weights for disjoint source.
                if abs(average[3]) < 1e-12:
                    average[:] = 0
                else:
                    average[:3] *= 255./average[3]
            mean[row, col] = average
            fractions[row, col] = inside_area/area
    pixels = np.clip(np.rint(mean), 0, 255).astype(np.uint8)
    if source.shape[2] == 4:
        pixels[..., :3][pixels[..., 3] == 0] = 0
    return pixels, mean, fractions


class IndependentAxisAreaTests(unittest.TestCase):
    def test_01_scalar_rectangle_clipping_rgb_rgba_borders_flips(self):
        random = np.random.default_rng(429201)
        maximum = 0.
        checked = 0
        for channels in [3, 4]:
            source = random.integers(0, 256, (7, 9, channels), dtype=np.uint8)
            if channels == 4:
                source[0, :, 3] = 0
            for scale in [[.2, .3], [1.7, .8], [-1.3, .6], [-2.1, -1.8]]:
                matrix = np.diag(scale)
                offset = np.array([2.13, 3.27])
                grid = OutputGrid((4, 5), (-1.1, -.9), (.73, .91))
                control = AxisAlignedAreaSampler(source, matrix, offset, fill_rgb=(17., 61., 203.))
                got, mean, fraction = control.sample(grid, max_tile_pixels=3)
                expected, ref_mean, ref_fraction = rectangle_reference(source, matrix, offset, grid, (17., 61., 203.))
                np.testing.assert_allclose(mean, ref_mean, atol=2e-8, rtol=2e-11)
                np.testing.assert_allclose(fraction, ref_fraction, atol=2e-14, rtol=2e-14)
                np.testing.assert_array_equal(got, expected)
                maximum = max(maximum, float(np.max(np.abs(mean-ref_mean))))
                checked += 1
        NOTES["scalar_rectangle_clipping"] = {"outputs": checked, "output_pixels": checked*20,
            "maximum_encoded_mean_difference": maximum, "uint8_pixel_mismatches": 0,
            "model": "Exact integral of encoded source-cell constant reconstruction, not unknown optical ink."}

    def test_02_known_32to1_counterexample_positive_control(self):
        source = np.full((65, 65, 3), 255, np.uint8)
        source[:, 32] = 0
        control = AxisAlignedAreaSampler(source, np.diag([32., 32.]), [32., 32.])
        image, mean, fraction = control.sample(OutputGrid((1, 1), (0., 0.), (1., 1.)))
        np.testing.assert_array_equal(image, 247)
        np.testing.assert_allclose(mean, 247.03125, atol=0, rtol=0)
        np.testing.assert_array_equal(fraction, 1.)
        NOTES["retained_counterexample_positive_control"] = {"encoded_mean": mean[0, 0].tolist(),
            "uint8": image[0, 0].tolist(), "scope": "Diagonal affine32:1 and source-cell model only."}

    def test_03_tile_partition_exact_and_full_source_identity(self):
        random = np.random.default_rng(429203)
        source = random.integers(0, 256, (11, 13, 3), dtype=np.uint8)
        grid = OutputGrid((7, 11), (-.2, -.1), (.7, .8))
        control = AxisAlignedAreaSampler(source, np.diag([1.3, -1.1]), [.17, 8.91])
        reference = control.sample(grid, max_tile_pixels=77)
        for budget in [1, 7, 19]:
            for a, b in zip(control.sample(grid, max_tile_pixels=budget), reference):
                np.testing.assert_array_equal(a, b)
        identical = AxisAlignedAreaSampler(source, np.eye(2), [0., 0.]).sample(
            OutputGrid(source.shape[:2], (0., 0.), (1., 1.)))
        np.testing.assert_array_equal(identical[0], source)

    def test_04_explicitly_rejects_off_diagonal_singular_nonfinite(self):
        source = np.zeros((3, 3, 3), np.uint8)
        for matrix in [np.array([[1., 1e-200], [0., 1.]]),
                       np.array([[0., 1.], [-1., 0.]]), np.diag([0., 1.]),
                       np.diag([np.inf, 1.]), np.diag([np.nan, 1.])]:
            with self.assertRaises(ValueError):
                AxisAlignedAreaSampler(source, matrix, [0., 0.])

    def test_05_rejects_unresolved_tiny_footprint(self):
        source = np.zeros((100, 100, 3), np.uint8)
        control = AxisAlignedAreaSampler(source, np.diag([1e-8, 1e-8]), [50., 50.])
        with self.assertRaises(ValueError):
            control.sample(OutputGrid((1, 1), (0., 0.), (1., 1.)))


if __name__ == "__main__":
    paths = [ROOT/"m4/sampling/axis_aligned_area.py", ROOT/"m4/sampling/axis_area_protocol.json", Path(__file__)]
    before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentAxisAreaTests))
    stable = all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == value for path, value in before.items())
    receipt = {"schema": "m4-independent-axis-area-tests-v1", "tests": result.testsRun,
        "passed": result.wasSuccessful() and stable, "sources_stable_during_run": stable,
        "source_sha256": before, "failures": [{"test": str(t), "trace": text} for t, text in result.failures],
        "errors": [{"test": str(t), "trace": text} for t, text in result.errors], "notes": NOTES,
        "limits": ["Failure-directed development control", "Diagonal affine maps only",
            "Different source reconstruction from bilinear candidate", "No new native ECG or clinical assessment"]}
    target = HERE/"receipts/independent-axis-area-tests.json"
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"receipt": str(target), "tests": result.testsRun, "passed": receipt["passed"]}))
    raise SystemExit(0 if receipt["passed"] else 1)
