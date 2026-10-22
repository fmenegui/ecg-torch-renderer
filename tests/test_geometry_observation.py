import unittest

import numpy as np

from ecg_gpu.digitization.observation import remove_periodic_banding


class ObservedBandTests(unittest.TestCase):
    def image(self):
        rgb = np.full((192, 320, 3), 60., float)
        rgb[30:162, 95:225] = 230
        for y in range(42, 155, 12):
            rgb[y, 100:220] = 165
        for x in range(107, 220, 12):
            rgb[35:157, x] = 165
        y, x = np.mgrid[42:155:12, 107:220:12]
        return rgb, np.column_stack((x.ravel(), y.ravel())).astype(float)

    def test_background_sinusoid_is_removed_without_using_paper_grid_rows(self):
        rgb, points = self.image()
        wave = 10*np.sin(2*np.pi*.073*np.arange(len(rgb)))
        observed = (rgb+wave[:, None, None]).round().clip(0, 255).astype(np.uint8)
        filtered, report = remove_periodic_banding(observed, points)
        self.assertTrue(report['applied'])
        self.assertLess(np.mean(np.abs(filtered.astype(float)-rgb)), .5)
        self.assertFalse(report['ground_truth_used'])

    def test_printed_horizontal_grid_does_not_trigger_background_filter(self):
        rgb, points = self.image()
        rgb = rgb.astype(np.uint8)
        filtered, report = remove_periodic_banding(rgb, points)
        self.assertFalse(report['applied'])
        np.testing.assert_array_equal(filtered, rgb)


if __name__ == '__main__':
    unittest.main()
