"""Offline numerical and failure-path controls for the M4 component."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image, ImageFile

import pngio
from streaming_sampler import OutputGrid, sample_image, iter_area_tiles, SamplingStats


def independent_bilinear(source, grid, n, inverse, fill=(255, 255, 255)):
    """Scalar four-neighbour reference for exact midpoint coordinates here.

    This uses no cv2.remap and stores no precomputed maps. The controls have
    exactly representable fractions on OpenCV's 1/32 interpolation table.
    """
    result = np.empty((*grid.shape_hw, 3), dtype=np.uint8)
    h, w = source.shape[:2]
    for row in range(grid.shape_hw[0]):
        for col in range(grid.shape_hw[1]):
            value = np.zeros(3, dtype=np.float64)
            for sy in range(n):
                for sx in range(n):
                    point = np.array([[grid.origin_mm[0]+col*grid.spacing_mm[0]+
                                       ((sx+.5)/n-.5)*grid.spacing_mm[0],
                                       grid.origin_mm[1]+row*grid.spacing_mm[1]+
                                       ((sy+.5)/n-.5)*grid.spacing_mm[1]]])
                    x, y = inverse(point)[0]
                    if not np.isfinite([x, y]).all():
                        value += fill
                        continue
                    x0, y0 = int(np.floor(x)), int(np.floor(y))
                    dx, dy = x-x0, y-y0
                    for ix, wx in ((x0, 1-dx), (x0+1, dx)):
                        for iy, wy in ((y0, 1-dy), (y0+1, dy)):
                            colour = source[iy, ix] if 0 <= ix < w and 0 <= iy < h else fill
                            value += np.asarray(colour)*wx*wy
            result[row, col] = np.clip(np.rint(value/(n*n)), 0, 255)
    return result


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)

    def setUp(self):
        self.rng = np.random.default_rng(4201)

    def test_constant_interior_all_arms_and_anisotropy(self):
        source = np.broadcast_to(np.array([47, 128, 219], np.uint8), (40, 55, 3)).copy()
        grid = OutputGrid((7, 9), (2., 3.), (.31, .17))
        inverse = lambda points: points @ np.array([[1.7, .3], [-.2, 1.1]]).T + [5., 4.]
        for n in (2, 4, 8):
            for reconstruction in ("bilinear", "boxhold"):
                with self.subTest(n=n, kind=reconstruction):
                    sampled = sample_image(source, inverse, grid, subgrid=n,
                                           reconstruction=reconstruction, max_tile_pixels=13)
                    np.testing.assert_array_equal(sampled.image, np.broadcast_to(source[0, 0], (7, 9, 3)))
                    self.assertTrue(np.all(sampled.full_reconstruction_support_fraction == 1))
                    self.assertTrue(np.all(sampled.finite_geometry_fraction == 1))
                    self.assertIsNone(sampled.trusted_geometry_fraction)

    def test_identity_nearest_including_edges(self):
        source = self.rng.integers(0, 256, (9, 11, 3), dtype=np.uint8)
        grid = OutputGrid((9, 11), (0., 0.), (1., 1.))
        for n in (2, 4, 8):
            sampled = sample_image(source, lambda p: p, grid, subgrid=n,
                                   reconstruction="boxhold", max_tile_pixels=7)
            np.testing.assert_array_equal(sampled.image, source)
            self.assertTrue(np.all(sampled.source_cell_fraction == 1))
            self.assertTrue(np.all(sampled.full_reconstruction_support_fraction == 1))

    def test_identity_bilinear_scalar_reference_and_edge_fill(self):
        source = self.rng.integers(0, 256, (5, 6, 3), dtype=np.uint8)
        grid = OutputGrid((5, 6), (0., 0.), (1., 1.))
        for n in (2, 4, 8):
            sampled = sample_image(source, lambda p: p, grid, subgrid=n, max_tile_pixels=7)
            expected = independent_bilinear(source, grid, n, lambda p: p)
            np.testing.assert_array_equal(sampled.image, expected)
            self.assertFalse(np.array_equal(sampled.image, source))
            self.assertTrue(np.all(sampled.source_cell_fraction == 1))
            self.assertEqual(sampled.full_reconstruction_support_fraction[0, 0], .25)

    def test_tile_invariance_rectangles_anisotropic_and_partial_tiles(self):
        source = self.rng.integers(0, 256, (17, 21, 3), dtype=np.uint8)
        grid = OutputGrid((11, 15), (-1., -.5), (.5, .25))
        inverse = lambda p: p @ np.array([[1.5, .25], [.125, 1.25]]).T + [1., 2.]
        for n in (2, 4, 8):
            for reconstruction in ("bilinear", "boxhold"):
                whole = sample_image(source, inverse, grid, subgrid=n,
                                     reconstruction=reconstruction, max_tile_pixels=1000)
                for budget in (1, 7, 16, 51):
                    with self.subTest(n=n, kind=reconstruction, tile=budget):
                        tiled = sample_image(source, inverse, grid, subgrid=n,
                                             reconstruction=reconstruction, max_tile_pixels=budget)
                        np.testing.assert_array_equal(tiled.image, whole.image)
                        np.testing.assert_array_equal(tiled.source_cell_fraction, whole.source_cell_fraction)
                        np.testing.assert_array_equal(tiled.full_reconstruction_support_fraction,
                                                      whole.full_reconstruction_support_fraction)
                        self.assertLessEqual(tiled.stats.max_query_points, budget)
                        self.assertLessEqual(tiled.stats.max_coordinate_bytes, budget*56)

    def test_invalid_no_valid_sentinel_or_warning(self):
        source = np.zeros((5, 5, 3), dtype=np.uint8)
        source[0, 0] = [210, 44, 13]
        grid = OutputGrid((2, 3), (0., 0.), (1., 1.))
        for value in ([np.nan, 0], [0, np.inf], [-np.inf, 2], [1e300, 1e300], [-1e300, 0]):
            for kind in ("bilinear", "boxhold"):
                with self.subTest(value=value, kind=kind), np.errstate(all="raise"):
                    sampled = sample_image(source, lambda p: np.tile(value, (len(p), 1)), grid,
                                           reconstruction=kind, max_tile_pixels=2,
                                           fill_rgb=(7, 8, 9))
                    np.testing.assert_array_equal(sampled.image, np.tile([7, 8, 9], (2, 3, 1)))
                    self.assertTrue(np.all(sampled.source_cell_fraction == 0))
                    self.assertTrue(np.all(sampled.full_reconstruction_support_fraction == 0))
                    self.assertTrue(np.all(sampled.finite_geometry_fraction == int(np.isfinite(value).all())))

    def test_partial_border_support_does_not_gate_rgb(self):
        source = np.zeros((3, 3, 3), dtype=np.uint8)
        grid = OutputGrid((1, 1), (0., 0.), (1., 1.))
        sampled = sample_image(source, lambda p: p, grid, subgrid=2)
        np.testing.assert_array_equal(sampled.image[0, 0], [60, 60, 60])
        self.assertEqual(sampled.source_cell_fraction[0, 0], 1)
        self.assertEqual(sampled.full_reconstruction_support_fraction[0, 0], .25)
        self.assertEqual(sampled.finite_geometry_fraction[0, 0], 1)

    def test_geometry_trust_has_no_effect_on_rgb_or_rgba(self):
        grid = OutputGrid((7, 8), (0., 0.), (1., 1.))
        for channels in (3, 4):
            source = self.rng.integers(0, 256, (7, 8, channels), dtype=np.uint8)
            unknown = sample_image(source, lambda p: p, grid, subgrid=4)
            untrusted = sample_image(source, lambda p: p, grid, subgrid=4,
                                     trust=lambda p: np.zeros(len(p), bool))
            np.testing.assert_array_equal(unknown.image, untrusted.image)
            self.assertTrue(np.all(untrusted.trusted_geometry_fraction == 0))

    def test_rgba_hidden_colours_do_not_bleed(self):
        first = np.zeros((3, 3, 4), dtype=np.uint8)
        first[1, 1] = [255, 0, 0, 255]
        second = first.copy()
        second[second[..., 3] == 0, :3] = [0, 255, 255]
        grid = OutputGrid((3, 3), (0., 0.), (1., 1.))
        a = sample_image(first, lambda p: p, grid, subgrid=8, max_tile_pixels=2)
        b = sample_image(second, lambda p: p, grid, subgrid=8, max_tile_pixels=9)
        np.testing.assert_array_equal(a.image, b.image)
        nontransparent = a.image[..., 3] != 0
        self.assertTrue(np.all(a.image[..., :3][nontransparent] == [255, 0, 0]))
        self.assertTrue(np.all(a.image[..., :3][~nontransparent] == 0))
        self.assertTrue(np.any(a.image[..., 3] < 255))

    def test_thin_trace_and_disclosed_bilinear_blur(self):
        source = np.full((31, 31, 3), 255, np.uint8)
        source[:, 11] = 0
        grid = OutputGrid((7, 13), (0., .5), (.25, .25))
        # Source 10 px/mm, output 4 px/mm. Point sampling misses x=11.
        points = np.arange(13)*2.5
        self.assertFalse(np.any(np.abs(points-11) <= .5))
        for n in (2, 4, 8):
            sampled = sample_image(source, lambda p: p*10, grid, subgrid=n)
            self.assertTrue(np.any(sampled.image < 255))
            self.assertGreater(int(sampled.image.min()), 0)
        identity = sample_image(source, lambda p: p,
                                OutputGrid((31, 31), (0., 0.), (1., 1.)), subgrid=8)
        self.assertGreater(int(identity.image[15, 11, 0]), 0)

    def test_iterator_does_not_allocate_a_collected_output(self):
        source = np.full((10, 10, 3), 128, np.uint8)
        stats = SamplingStats()
        tiles = iter_area_tiles(source, lambda p: p,
                                OutputGrid((23, 37), (0., 0.), (.1, .1)),
                                subgrid=8, max_tile_pixels=17, stats=stats)
        total = 0
        for tile in tiles:
            total += tile.image.shape[0]*tile.image.shape[1]
            self.assertLessEqual(tile.image.shape[0]*tile.image.shape[1], 17)
        self.assertEqual(total, 23*37)
        self.assertLessEqual(stats.max_query_points, 17)
        self.assertEqual(stats.inverse_points, 23*37*64)

    def test_bad_arguments_and_callbacks(self):
        source = np.zeros((2, 2, 3), np.uint8)
        grid = OutputGrid((2, 2), (0., 0.), (1., 1.))
        for kwargs in ({"subgrid": 3}, {"subgrid": True}, {"subgrid": 2.0}, {"max_tile_pixels": 0},
                       {"max_tile_pixels": 2.5}, {"reconstruction": "cubic"}):
            with self.assertRaises(ValueError):
                sample_image(source, lambda p: p, grid, **kwargs)
        with self.assertRaises(ValueError):
            sample_image(source, lambda p: np.zeros((len(p), 3)), grid)
        with self.assertRaises(ValueError):
            sample_image(source, lambda p: p, grid, trust=lambda p: np.ones(len(p)))
        with self.assertRaises(ValueError):
            OutputGrid((0, 2), (0., 0.), (1., 1.))
        with self.assertRaises(ValueError):
            OutputGrid((2, 2), (0., 0.), (-1., 1.))
        for source_bad in (None, np.array(1), np.zeros((2, 2, 2), np.uint8)):
            with self.assertRaises(ValueError):
                sample_image(source_bad, lambda p: p, grid)


class PngTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.path = self.directory/"output.png"
        self.rgb = np.random.default_rng(18).integers(0, 256, (19, 23, 3), np.uint8)

    def tearDown(self):
        self.temp.cleanup()

    def test_rgb_rgba_verified_roundtrip(self):
        for channels in (3, 4):
            path = self.directory/f"output-{channels}.png"
            image = np.concatenate((self.rgb, self.rgb[..., :1]), -1) if channels == 4 else self.rgb
            receipt = pngio.write_png_atomic(path, image)
            self.assertEqual(receipt.decoded_sha256, hashlib.sha256(image.tobytes()).hexdigest())
            pngio.verify_png(path, image)
            self.assertEqual(receipt.encoded_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertFalse(receipt.replaced_existing)
        self.assertEqual(len(list(self.directory.iterdir())), 2)

    def test_default_refuses_existing_and_guard_allows_explicit_replace(self):
        first = pngio.write_png_atomic(self.path, self.rgb)
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            pngio.write_png_atomic(self.path, 255-self.rgb)
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises(ValueError):
            pngio.write_png_atomic(self.path, 255-self.rgb, expected_existing_sha256="0"*64)
        self.assertEqual(self.path.read_bytes(), before)
        receipt = pngio.write_png_atomic(self.path, 255-self.rgb,
                                         expected_existing_sha256=first.encoded_sha256)
        self.assertTrue(receipt.replaced_existing)
        pngio.verify_png(self.path, 255-self.rgb)
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_truncated_encoded_temp_never_published(self):
        real_save = Image.Image.save
        def truncated_save(image, stream, **kwargs):
            real_save(image, stream, **kwargs)
            stream.truncate(max(8, stream.tell()-15))
        with mock.patch.object(Image.Image, "save", truncated_save):
            with self.assertRaises(ValueError):
                pngio.write_png_atomic(self.path, self.rgb)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_failed_decode_keeps_existing_file(self):
        receipt = pngio.write_png_atomic(self.path, self.rgb)
        before = self.path.read_bytes()
        with mock.patch.object(pngio, "verify_png", side_effect=ValueError("decode failure")):
            with self.assertRaises(ValueError):
                pngio.write_png_atomic(self.path, 255-self.rgb,
                                         expected_existing_sha256=receipt.encoded_sha256)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_atomic_no_clobber_race(self):
        real_link = pngio.os.link
        def contender(source, destination):
            Path(destination).write_bytes(b"external winner")
            return real_link(source, destination)
        with mock.patch.object(pngio.os, "link", contender):
            with self.assertRaises(FileExistsError):
                pngio.write_png_atomic(self.path, self.rgb)
        self.assertEqual(self.path.read_bytes(), b"external winner")
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_guard_rechecked_before_replace(self):
        receipt = pngio.write_png_atomic(self.path, self.rgb)
        real_verify = pngio.verify_png
        def contender(path, expected):
            result = real_verify(path, expected)
            self.path.write_bytes(b"external change")
            return result
        with mock.patch.object(pngio, "verify_png", contender):
            with self.assertRaises(ValueError):
                pngio.write_png_atomic(self.path, 255-self.rgb,
                                         expected_existing_sha256=receipt.encoded_sha256)
        self.assertEqual(self.path.read_bytes(), b"external change")
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_lock_is_not_stolen_and_invalid_inputs_do_not_write(self):
        lock = self.directory/".output.png.write-lock"
        lock.write_text("another writer")
        with self.assertRaises(FileExistsError):
            pngio.write_png_atomic(self.path, self.rgb)
        self.assertEqual(lock.read_text(), "another writer")
        lock.unlink()
        with self.assertRaises(ValueError):
            pngio.write_png_atomic(self.directory/"output.jpg", self.rgb)
        with self.assertRaises(ValueError):
            pngio.write_png_atomic(self.path, self.rgb.astype(float))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_permissive_decoder_setting_fails_closed(self):
        with mock.patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True):
            with self.assertRaises(ValueError):
                pngio.write_png_atomic(self.path, self.rgb)
        self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
