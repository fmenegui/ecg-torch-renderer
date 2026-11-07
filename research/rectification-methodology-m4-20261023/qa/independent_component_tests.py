"""Independent finite adversarial checks. No neural inference or frozen edits.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
swarm-venv/bin/python m4/qa/independent_component_tests.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "m4/sampling"))
sys.path.insert(0, str(ROOT / "m4/geometry"))
import streaming_sampler as sampler
import pngio
import geometry_selector as geometry

NOTES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct_project(points, matrix):
    """Independent explicit scalar homography evaluation."""
    answer = []
    for x, y in np.asarray(points):
        d = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        answer.append([(matrix[0, 0]*x + matrix[0, 1]*y + matrix[0, 2])/d,
                       (matrix[1, 0]*x + matrix[1, 1]*y + matrix[1, 2])/d])
    return np.asarray(answer)


def scalar_sample(source, x, y, mode, fill):
    """No cv2 call. Reconstruct using the documented quantized coordinates."""
    h, w, c = source.shape
    if not np.isfinite([x, y]).all():
        return np.asarray(fill, np.float32)
    if x < -1 or x > w or y < -1 or y > h:
        return np.asarray(fill, np.float32)
    x, y = float(np.float32(x)), float(np.float32(y))
    def value(a, b):
        return source[b, a] if 0 <= a < w and 0 <= b < h else fill
    if mode == "boxhold":
        return np.asarray(value(int(np.rint(x)), int(np.rint(y))), np.float32)
    # OpenCV INTER_LINEAR has 32 fractional positions per source pixel.
    x, y = np.rint(x*32)/32, np.rint(y*32)/32
    left, top = int(np.floor(x)), int(np.floor(y))
    fx, fy = x-left, y-top
    answer = np.zeros(c, float)
    for a, wx in [(left, 1-fx), (left+1, fx)]:
        for b, wy in [(top, 1-fy), (top+1, fy)]:
            answer += np.asarray(value(a, b))*wx*wy
    return answer.astype(np.float32)


def scalar_quadrature(source_u8, inverse, grid, n, mode, fill=(255., 255., 255.)):
    source = source_u8.astype(np.float32)
    if source.shape[2] == 4:
        source[..., :3] *= source[..., 3:4]/np.float32(255)
        fill = (0., 0., 0., 0.)
    image = np.empty((*grid.shape_hw, source.shape[2]), np.uint8)
    for row in range(grid.shape_hw[0]):
        for col in range(grid.shape_hw[1]):
            total = np.zeros(source.shape[2], np.float32)
            for ky in range(n):
                for kx in range(n):
                    px = grid.origin_mm[0]+col*grid.spacing_mm[0]+((kx+.5)/n-.5)*grid.spacing_mm[0]
                    py = grid.origin_mm[1]+row*grid.spacing_mm[1]+((ky+.5)/n-.5)*grid.spacing_mm[1]
                    xy = inverse(np.array([[px, py]], float))[0]
                    total += scalar_sample(source, *xy, mode, fill)
            total /= np.float32(n*n)
            if source.shape[2] == 4:
                total[:3] = total[:3]*np.float32(255)/total[3] if total[3] > 0 else 0
            result = np.clip(np.rint(total), 0, 255).astype(np.uint8)
            if len(result) == 4 and result[3] == 0:
                result[:3] = 0
            image[row, col] = result
    return image


class SamplingTests(unittest.TestCase):
    def test_01_scalar_oracle_rgb_and_rgba(self):
        rng = np.random.default_rng(42901)
        grid = sampler.OutputGrid((4, 5), (-.3, -.1), (.81, .73))
        inverse = lambda q: np.c_[1.07*q[:, 0]+.21*q[:, 1]+.13,
                                  -.09*q[:, 0]+.93*q[:, 1]+.47]
        max_difference = 0
        for channels in [3, 4]:
            image = rng.integers(0, 256, (7, 8, channels), dtype=np.uint8)
            for n in [2, 4, 8]:
                for mode in ["bilinear", "boxhold"]:
                    got = sampler.sample_image(image, inverse, grid, subgrid=n,
                                               reconstruction=mode, max_tile_pixels=7)
                    expected = scalar_quadrature(image, inverse, grid, n, mode)
                    difference = int(np.max(np.abs(got.image.astype(int)-expected.astype(int))))
                    max_difference = max(max_difference, difference)
                    self.assertLessEqual(difference, 1, (channels, n, mode))
        NOTES["scalar_oracle"] = {"outputs": 12, "max_uint8_difference": max_difference,
            "tolerance_reason": "Independent scalar arithmetic can differ at final half-LSB ties. No cv2 reference sampling call."}

    def test_02_trust_does_not_change_observed_rgb_or_alpha(self):
        rng = np.random.default_rng(42902)
        grid = sampler.OutputGrid((7, 9), (1., 1.), (.8, .7))
        inverse = lambda q: q + [1., 1.]
        for channels in [3, 4]:
            image = rng.integers(0, 256, (12, 13, channels), dtype=np.uint8)
            plain = sampler.sample_image(image, inverse, grid, subgrid=4)
            for trust in [lambda q: np.zeros(len(q), bool),
                          lambda q: np.ones(len(q), bool),
                          lambda q: q[:, 0] > 3]:
                got = sampler.sample_image(image, inverse, grid, subgrid=4, trust=trust)
                np.testing.assert_array_equal(got.image, plain.image)
                self.assertIsNotNone(got.trusted_geometry_fraction)
            zero = sampler.sample_image(image, inverse, grid, trust=lambda q: np.zeros(len(q), bool))
            self.assertEqual(float(zero.trusted_geometry_fraction.max()), 0.)

    def test_03_fully_supported_constant_and_box_identity(self):
        color = [12, 73, 211]
        image = np.broadcast_to(np.array(color, np.uint8), (11, 13, 3)).copy()
        interior = sampler.OutputGrid((4, 5), (2., 2.), (1.2, 1.1))
        for n in [2, 4, 8]:
            for mode in ["bilinear", "boxhold"]:
                got = sampler.sample_image(image, lambda q: q, interior,
                    subgrid=n, reconstruction=mode, max_tile_pixels=3)
                np.testing.assert_array_equal(got.image, np.broadcast_to(color, got.image.shape))
                np.testing.assert_array_equal(got.full_reconstruction_support_fraction, 1.)
        rng = np.random.default_rng(42903)
        image = rng.integers(0, 256, (8, 9, 3), dtype=np.uint8)
        got = sampler.sample_image(image, lambda q: q,
            sampler.OutputGrid(image.shape[:2], (0., 0.), (1., 1.)), reconstruction="boxhold")
        np.testing.assert_array_equal(got.image, image)

    def test_04_nonfinite_and_extreme_do_not_sample_origin(self):
        image = np.full((5, 5, 3), 255, np.uint8)
        image[0, 0] = [0, 17, 85]
        grid = sampler.OutputGrid((2, 3), (0., 0.), (1., 1.))
        for point in [[np.nan, 0.], [np.inf, -np.inf], [1e300, 1e300], [-1e300, 0.]]:
            inverse = lambda q, p=point: np.broadcast_to(p, q.shape).copy()
            got = sampler.sample_image(image, inverse, grid, subgrid=8,
                trust=lambda q: np.ones(len(q), bool))
            np.testing.assert_array_equal(got.image, 255)
            np.testing.assert_array_equal(got.source_cell_fraction, 0.)
            np.testing.assert_array_equal(got.full_reconstruction_support_fraction, 0.)
            expected_finite = float(np.isfinite(point).all())
            np.testing.assert_array_equal(got.finite_geometry_fraction, expected_finite)
            np.testing.assert_array_equal(got.trusted_geometry_fraction, expected_finite)

    def test_05_border_fraction_and_no_renormalization(self):
        image = np.zeros((3, 3, 3), np.uint8)
        grid = sampler.OutputGrid((1, 1), (0., 0.), (1., 1.))
        inverse = lambda q: q + [-.5, 1.]
        got = sampler.sample_image(image, inverse, grid, subgrid=2)
        self.assertEqual(float(got.finite_geometry_fraction[0, 0]), 1.)
        self.assertEqual(float(got.source_cell_fraction[0, 0]), .5)
        self.assertEqual(float(got.full_reconstruction_support_fraction[0, 0]), 0.)
        # x samples are -.75 and -.25 with white-padding weights .75,.25.
        np.testing.assert_array_equal(got.image[0, 0], [128, 128, 128])
        for x, inside in [(-.5, 1.), (2.5, 0.)]:
            inv = lambda q, x=x: np.broadcast_to([x, 1.], q.shape).copy()
            got = sampler.sample_image(image, inv, grid, subgrid=2)
            self.assertEqual(float(got.source_cell_fraction[0, 0]), inside)

    def test_06_rgba_hidden_color_and_partial_alpha(self):
        source = np.array([[[0, 255, 0, 0], [255, 0, 0, 255]]], np.uint8)
        grid = sampler.OutputGrid((1, 1), (0., 0.), (1., 1.))
        inverse = lambda q: np.broadcast_to([.5, 0.], q.shape).copy()
        got = sampler.sample_image(source, inverse, grid, subgrid=4)
        np.testing.assert_array_equal(got.image[0, 0], [255, 0, 0, 128])
        source[0, 0, :3] = [73, 12, 241]
        changed = sampler.sample_image(source, inverse, grid, subgrid=4)
        np.testing.assert_array_equal(changed.image, got.image)

    def test_07_tile_size_and_fraction_parity(self):
        rng = np.random.default_rng(42907)
        source = rng.integers(0, 256, (13, 17, 4), dtype=np.uint8)
        grid = sampler.OutputGrid((9, 19), (-1., 0.), (.8, 1.1))
        inverse = lambda q: np.c_[q[:, 0]+.12*np.sin(q[:, 1]), q[:, 1]+.03*q[:, 0]**2]
        trust = lambda q: q[:, 0]+q[:, 1] > 8
        for n in [2, 4, 8]:
            for mode in ["bilinear", "boxhold"]:
                reference = sampler.sample_image(source, inverse, grid, subgrid=n,
                    reconstruction=mode, max_tile_pixels=171, trust=trust)
                for budget in [1, 7, 23, 50]:
                    got = sampler.sample_image(source, inverse, grid, subgrid=n,
                        reconstruction=mode, max_tile_pixels=budget, trust=trust)
                    for key in ["image", "finite_geometry_fraction", "source_cell_fraction",
                                "full_reconstruction_support_fraction", "trusted_geometry_fraction"]:
                        np.testing.assert_array_equal(getattr(got, key), getattr(reference, key))
                    self.assertLessEqual(got.stats.max_query_points, budget)
                    self.assertEqual(got.stats.inverse_points, 9*19*n*n)

    def test_08_explicit_coordinate_work_is_tile_bounded(self):
        source = np.zeros((3, 3, 3), np.uint8)
        seen = []
        def inverse(q):
            seen.append(len(q))
            return np.broadcast_to([1., 1.], q.shape).copy()
        stats = sampler.SamplingStats()
        digest = hashlib.sha256()
        for tile in sampler.iter_area_tiles(source, inverse,
                sampler.OutputGrid((37, 53), (0., 0.), (1., 1.)), subgrid=8,
                max_tile_pixels=31, stats=stats):
            digest.update(tile.image.tobytes())
        self.assertLessEqual(max(seen), 31)
        self.assertEqual(stats.inverse_points, 37*53*64)
        self.assertLessEqual(stats.max_coordinate_bytes, 31*56)
        NOTES["streaming"] = {"max_query_points": max(seen), "points": stats.inverse_points,
            "max_explicit_coordinate_bytes": stats.max_coordinate_bytes,
            "scope": "Explicit owned coordinate arrays only, excludes callbacks/allocator/source/output."}

    def test_09_finite_rules_can_all_erase_observed_trace(self):
        source = np.full((65, 65, 3), 255, np.uint8)
        source[:, 32] = 0
        grid = sampler.OutputGrid((1, 1), (0., 0.), (1., 1.))
        inverse = lambda q: 32.+32.*q
        outputs = {}
        for n in [2, 4, 8]:
            got = sampler.sample_image(source, inverse, grid, subgrid=n)
            outputs[str(n)] = got.image[0, 0].tolist()
            np.testing.assert_array_equal(got.image[0, 0], [255, 255, 255])
            np.testing.assert_array_equal(got.full_reconstruction_support_fraction, 1.)
        # Bilinear dark tent of height255 and base2 source pixels has area255.
        exact_mean = 255.-255./32.
        self.assertEqual(exact_mean, 247.03125)
        NOTES["retained_alias_counterexample"] = {"source": "65x65 white, column32 black",
            "inverse": "source_xy=32+32*physical_xy, one output pixel of spacing1",
            "outputs": outputs, "exact_encoded_bilinear_area_mean": exact_mean,
            "exact_boxhold_area_mean": exact_mean,
            "consequence": "Agreement of2/4/8 finite quadratures is not an accuracy certificate. All erase an observed stroke at32x minification."}

    def test_10_invalid_contracts_raise(self):
        source = np.zeros((3, 3, 3), np.uint8)
        grid = sampler.OutputGrid((1, 1), (0., 0.), (1., 1.))
        for n in [True, 0, 3, 2.5]:
            with self.assertRaises((ValueError, TypeError)):
                sampler.sample_image(source, lambda q: q, grid, subgrid=n)
        with self.assertRaises(ValueError):
            sampler.sample_image(source, lambda q: np.zeros((len(q), 3)), grid)
        with self.assertRaises(ValueError):
            sampler.sample_image(source, lambda q: q, grid, trust=lambda q: np.ones(len(q)))
        with self.assertRaises(ValueError):
            sampler.sample_image(np.zeros((3, 3, 1), np.uint8), lambda q: q, grid)


class ExportTests(unittest.TestCase):
    def test_11_full_decode_rgb_rgba_and_no_clobber(self):
        with tempfile.TemporaryDirectory(dir=HERE / "receipts") as folder:
            rng = np.random.default_rng(42911)
            for channels in [3, 4]:
                pixels = rng.integers(0, 256, (13, 17, channels), dtype=np.uint8)
                path = Path(folder)/f"roundtrip-{channels}.png"
                receipt = pngio.write_png_atomic(path, pixels)
                self.assertEqual(receipt.encoded_sha256, sha(path))
                self.assertEqual(receipt.decoded_sha256, hashlib.sha256(pixels.tobytes()).hexdigest())
                with Image.open(path) as image:
                    image.load()
                    np.testing.assert_array_equal(np.array(image), pixels)
                old = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    pngio.write_png_atomic(path, 255-pixels)
                self.assertEqual(path.read_bytes(), old)
                new = pngio.write_png_atomic(path, 255-pixels,
                    expected_existing_sha256=receipt.encoded_sha256)
                self.assertTrue(new.replaced_existing)
                pngio.verify_png(path, 255-pixels)

    def test_12_corrupt_temporary_never_replaces_valid_target(self):
        with tempfile.TemporaryDirectory(dir=HERE / "receipts") as folder:
            pixels = np.zeros((11, 13, 3), np.uint8)
            path = Path(folder)/"target.png"
            pngio.write_png_atomic(path, pixels)
            previous = path.read_bytes()
            verifier = pngio.verify_png
            def corrupt_then_verify(temporary, expected):
                data = Path(temporary).read_bytes()
                Path(temporary).write_bytes(data[:-7])
                return verifier(temporary, expected)
            with mock.patch.object(pngio, "verify_png", side_effect=corrupt_then_verify):
                with self.assertRaises(ValueError):
                    pngio.write_png_atomic(path, 255-pixels, expected_existing_sha256=sha(path))
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["target.png"])

    def test_13_encoding_and_replace_errors_preserve_existing(self):
        with tempfile.TemporaryDirectory(dir=HERE / "receipts") as folder:
            pixels = np.zeros((3, 4, 4), np.uint8)
            path = Path(folder)/"target.png"
            pngio.write_png_atomic(path, pixels)
            old = path.read_bytes()
            for where in ["encode", "replace"]:
                patch = mock.patch.object(Image.Image, "save", side_effect=OSError("injected encode failure")) if where == "encode" else mock.patch.object(pngio.os, "replace", side_effect=OSError("injected replace failure"))
                with patch:
                    with self.assertRaises(OSError):
                        pngio.write_png_atomic(path, 255-pixels, expected_existing_sha256=sha(path))
                self.assertEqual(path.read_bytes(), old)
                self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["target.png"])

    def test_14_png_terminal_crc_pixels_and_permissive_decoder(self):
        with tempfile.TemporaryDirectory(dir=HERE / "receipts") as folder:
            pixels = np.zeros((3, 4, 3), np.uint8)
            original = Path(folder)/"good.png"
            pngio.write_png_atomic(original, pixels)
            data = original.read_bytes()
            for name, malformed in [("no-iend", data[:-12]), ("trailing", data+b"extra"),
                                    ("crc", data[:-1]+bytes([data[-1]^1]))]:
                path = Path(folder)/(name+".png")
                path.write_bytes(malformed)
                with self.assertRaises(ValueError):
                    pngio.verify_png(path, pixels)
            with self.assertRaises(ValueError):
                pngio.verify_png(original, 255-pixels)
            from PIL import ImageFile
            with mock.patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True):
                with self.assertRaises(ValueError):
                    pngio.verify_png(original, pixels)


class GeometryTests(unittest.TestCase):
    @staticmethod
    def regular(nr=8, nc=10):
        rr, cc = np.mgrid[:nr, :nc]
        ids = np.c_[rr.ravel(), cc.ravel()]
        return ids, ids[:, ::-1]*5., [(r, c) for r in range(nr-1) for c in range(nc-1)]

    def test_15_clean_projective_formula_and_roundtrip(self):
        ids, physical, allowed = self.regular()
        matrix = np.array([[10., .7, 40.], [.4, 9., 35.], [.035, .012, 1.]])
        observed = direct_project(physical, matrix)
        model = geometry.Geometry(observed, physical, ids, allowed, np.linalg.inv(matrix))
        rng = np.random.default_rng(42915)
        truth = rng.uniform([.1, .1], [44.9, 34.9], (400, 2))
        source = direct_project(truth, matrix)
        pred = model.forward(source)
        np.testing.assert_allclose(pred, truth, rtol=0, atol=2e-11)
        np.testing.assert_allclose(model.inverse(truth), source, rtol=0, atol=2e-10)
        fitted, _ = geometry.fit_h(observed, physical, "LS")
        np.testing.assert_allclose(direct_project(source, fitted), truth, rtol=0, atol=2e-5)

    def test_16_support_hole_and_exterior_are_not_bridged(self):
        ids, physical, allowed = self.regular()
        allowed = [cell for cell in allowed if cell != (3, 4)]
        model = geometry.Geometry(physical, physical, ids, allowed, np.eye(3))
        query = np.array([[22.5, 17.5], [-.01, 3.], [45.01, 15.], [5., 5.]])
        got, status = model.forward(query, True)
        self.assertTrue(np.isnan(got[:3]).all())
        np.testing.assert_allclose(got[3], query[3], rtol=0, atol=1e-12)
        self.assertEqual(status[0], "outside-observed-support")

    def test_17_horizon_and_malformed_ids_rejected(self):
        ids, physical, allowed = self.regular()
        bad = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., -20.]])
        with self.assertRaises(ValueError):
            geometry.guard_h(bad, physical)
        duplicate = ids.copy()
        duplicate[-1] = duplicate[0]
        with self.assertRaises(ValueError):
            geometry.lattice(duplicate)
        fractional = ids.astype(float)
        fractional[0, 0] = .5
        with self.assertRaises(ValueError):
            geometry.lattice(fractional)
        line = np.c_[np.arange(10.), np.zeros(10)]
        with self.assertRaises(ValueError):
            geometry.fit_h(line, line, "LS")

    def test_18_positive_area_double_wrap_rejected(self):
        rr, cc = np.mgrid[:2, :49]
        ids = np.c_[rr.ravel(), cc.ravel()]
        angle = ids[:, 1]*4*np.pi/48
        radius = 1+ids[:, 0]
        observed = np.c_[radius*np.cos(angle), radius*np.sin(angle)]
        triangles = geometry.cells.cell_triangles(observed, ids[:, ::-1]*5., ids)
        with self.assertRaises(ValueError):
            geometry.guard_complex(observed, triangles, "double-wrap")

    def test_19_non_simple_boundary_single_wrap_rejected(self):
        rr, cc = np.mgrid[:2, :25]
        ids = np.c_[rr.ravel(), cc.ravel()]
        angle = ids[:, 1]*2*np.pi/24
        radius = 1+ids[:, 0]
        observed = np.c_[radius*np.cos(angle), radius*np.sin(angle)]
        canonical = ids[:, ::-1]*5.
        triangles = geometry.cells.cell_triangles(observed, canonical, ids)
        with self.assertRaises(ValueError):
            geometry.guard_complex(observed, triangles, "single-wrap-boundary")

    def test_20_shared_residual_edges_and_inverse_limits(self):
        ids, canonical, allowed = self.regular()
        observed = canonical.copy()
        observed[:, 0] += .4*np.sin(canonical[:, 1]/12.)
        observed[:, 1] += .6*np.sin(canonical[:, 0]/15.)
        model = geometry.Geometry(observed, canonical, ids, allowed, np.eye(3), observed)
        edge_to_triangles = {}
        for t, triangle in enumerate(model.triangles):
            for a, b in [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])]:
                edge_to_triangles.setdefault(tuple(sorted((int(a), int(b)))), []).append(t)
        maximum = 0.
        checked = 0
        for edge, adjacent in edge_to_triangles.items():
            if len(adjacent) != 2:
                continue
            mid = observed[list(edge)].mean(0)
            target = canonical[list(edge)].mean(0)
            points = []
            for index in adjacent:
                centre = observed[model.triangles[index]].mean(0)
                points.append(mid+(centre-mid)*1e-7)
            limit = model.forward(np.asarray(points))
            maximum = max(maximum, float(np.linalg.norm(limit[0]-limit[1])))
            np.testing.assert_allclose(limit, np.broadcast_to(target, (2, 2)), rtol=0, atol=1e-5)
            np.testing.assert_allclose(model.inverse(target[None]), mid[None], rtol=0, atol=1e-10)
            checked += 1
        self.assertGreater(checked, 100)
        NOTES["shared_edges"] = {"checked": checked, "two_sided_epsilon_limit_max_mm": maximum,
            "epsilon": 1e-7, "scope": "Finite regular continuous residual construction only."}

    def test_21_no_gold_argument_and_explicit_unverified_identity(self):
        self.assertEqual(list(inspect.signature(geometry.fit_select).parameters),
                         ["observed", "ids", "allowed_cells", "pitch_mm"])
        source = inspect.getsource(geometry.fit_select)
        for forbidden in ["physical_queries", "reference_mm", "forward_reference", "results.json"]:
            self.assertNotIn(forbidden, source)
        self.assertIn("remain unverified", source)
        # A source inspection is evidence of interface isolation, not a formal
        # noninterference proof for arbitrary external callbacks or globals.


def main():
    (HERE / "receipts").mkdir(exist_ok=True)
    before = {str(path.relative_to(ROOT)): sha(path) for path in
              [Path(sampler.__file__), Path(pngio.__file__), Path(geometry.__file__), Path(__file__)]}
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {str(ROOT/path): sha(ROOT/path) for path in before}
    stable = all(before[path] == after[str(ROOT/path)] for path in before)
    receipt = {"schema": "m4-independent-component-tests-v1", "tests": result.testsRun,
        "failures": [{"test": str(t), "trace": text} for t, text in result.failures],
        "errors": [{"test": str(t), "trace": text} for t, text in result.errors],
        "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped],
        "passed": result.wasSuccessful() and stable, "component_sources_stable_during_run": stable,
        "source_sha256": before, "elapsed_seconds": time.perf_counter()-started,
        "notes": NOTES, "limits": ["Finite component checks only", "No neural rerun",
        "No real clinical image or commercial endpoint evaluation", "No claim of every possible input"]}
    path = HERE / "receipts/independent-component-tests.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"receipt": str(path), "tests": result.testsRun,
                      "passed": receipt["passed"], "sha256": sha(path)}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
