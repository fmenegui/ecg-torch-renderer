"""Independent optics, exposure, material-scale and integration acceptance tests."""

import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from ecg_gpu import (
    ECGRenderer,
    RenderConfig,
    ECGImageLayer,
    DocumentTemplate,
    DOCUMENT_TEMPLATES,
    ThermalPrint,
    ThermalPrintConfig,
    PhysicalCamera,
    LensConfig,
    DisplayConfig,
    EmissiveDisplay,
    MeasuredPaperBank,
    PaperScan,
    SensorConfig,
    MaterialHistory,
    SceneBank,
)
from ecg_gpu.gpu_appearance import CameraAppearance


class RealismAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **kwargs):
        return RenderConfig(
            **dict(
                width=192,
                height=144,
                profile="photo",
                severity=0.3,
                surface_mode="off",
                camera_mode="sensor",
                jpeg_enabled=False,
                jpeg_probability=0,
                **kwargs,
            )
        )

    def signal(self):
        t = torch.arange(1000) / 500
        return (
            torch.sin(t * 7)[None, None] * torch.linspace(0.1, 0.6, 12)[None, :, None]
        ).contiguous()

    def render(self, renderer, **kwargs):
        return renderer(
            self.signal(),
            500,
            generator=torch.Generator().manual_seed(71),
            stage="all",
            return_instances=True,
            return_field=True,
            **kwargs,
        )

    def identity_field(self, h=48, w=64):
        y, x = torch.meshgrid(
            (torch.arange(h) + 0.5) * (2 / h) - 1,
            (torch.arange(w) + 0.5) * (2 / w) - 1,
            indexing="ij",
        )
        return torch.stack([x, y], -1)[None]

    def test_thin_lens_matches_independent_image_plane_geometry(self):
        c = LensConfig(
            focal_length_mm=8.0,
            focus_distance_m=0.5,
            camera_distance_m=0.5,
            f_number=2.8,
            pixel_pitch_um=3.0,
        )
        camera = PhysicalCamera(c)
        depths = torch.tensor([0.22, 0.5, 0.83], dtype=torch.float64)
        f = c.focal_length_mm / 1000
        sensor_plane = 1 / (1 / f - 1 / c.focus_distance_m)
        expected = []
        for distance in depths.tolist():
            image_plane = 1 / (1 / f - 1 / distance)
            diameter = (f / c.f_number) * abs(image_plane - sensor_plane) / image_plane
            expected.append(diameter / (c.pixel_pitch_um * 1e-6))
        torch.testing.assert_close(
            camera.circle_of_confusion_pixels(depths),
            torch.tensor(expected, dtype=torch.float64),
        )

    def test_focus_is_identity_and_constant_radiance_is_preserved(self):
        camera = PhysicalCamera()
        image = torch.rand(2, 3, 32, 40)
        focus = torch.full((2, 1, 32, 40), camera.config.focus_distance_m)
        self.assertTrue(torch.equal(camera.depth_blur(image, focus), image))
        depth = torch.linspace(0.12, 0.8, 40)[None, None, None].expand(2, 1, 32, 40)
        torch.testing.assert_close(
            camera.depth_blur(torch.full_like(image, 0.37), depth),
            torch.full_like(image, 0.37),
        )

    def test_depth_bound_and_nonphysical_depth_are_rejected(self):
        camera = PhysicalCamera()
        image = torch.ones(1, 3, 8, 8)
        for value in (0.004, 0.01, float("nan")):
            with self.assertRaises(RuntimeError):
                camera.depth_blur(image, torch.full_like(image[:, :1], value))
        with self.assertRaises(ValueError):
            camera.depth_blur(image, torch.ones(1, 8, 8))

    def test_depth_blur_reduces_checkerboard_contrast(self):
        camera = PhysicalCamera()
        image = (torch.arange(64)[None, None, None] % 2).float().expand(1, 3, 48, 64)
        blurred = camera.depth_blur(image, torch.full_like(image[:, :1], 0.1))
        self.assertLess(float(blurred[..., 6:-6, 6:-6].std()), float(image.std()) * 0.5)

    def test_defocus_mixture_matches_blur_disk_second_moment(self):
        camera = PhysicalCamera()
        c = camera.config
        f = c.focal_length_mm * 0.001
        coefficient = (
            f * f / (c.f_number * (c.focus_distance_m - f) * c.pixel_pitch_um * 1e-6)
        )
        sigma = 1.5
        depth = c.focus_distance_m / (1 + 4 * sigma / coefficient)
        impulse = torch.zeros(1, 3, 65, 65)
        impulse[:, :, 32, 32] = 1
        blurred = camera.depth_blur(impulse, torch.full_like(impulse[:, :1], depth))
        x = torch.arange(65) - 32
        variance = (blurred[0, 0] * x[None].square()).sum()
        # Finite 3-sigma kernels retain small truncation/discretization error.
        self.assertAlmostEqual(float(variance), sigma**2, delta=0.03)
        self.assertAlmostEqual(float(blurred[0, 0].sum()), 1.0, places=5)

    def test_camera_appearance_does_not_change_reference_uv_or_targets(self):
        base = self.render(ECGRenderer(self.config()))
        optical = self.render(
            ECGRenderer(
                self.config(),
                camera_model=PhysicalCamera(
                    LensConfig(focus_distance_m=0.16), jpeg_enabled=False
                ),
            )
        )
        self.assertFalse(torch.equal(base.image, optical.image))
        self.assertTrue(torch.equal(base.reference, self.signal()))
        for key in base.targets:
            self.assertTrue(torch.equal(base.targets[key], optical.targets[key]), key)
        self.assertTrue(torch.equal(base.inverse_field, optical.inverse_field))

    def test_depth_conversion_uses_half_physical_sheet_width(self):
        camera = PhysicalCamera()
        valid = torch.ones(2, 1, 4, 4)
        scene = {"height": valid * 0.1, "paper_mask": valid}
        depth = camera.scene_depth(scene, torch.tensor([2.0, 4.0]), 200, valid)
        torch.testing.assert_close(depth[:, 0, 0, 0], torch.tensor([0.375, 0.3775]))

    def test_calibration_requires_explicit_measured_evidence(self):
        with self.assertRaises(ValueError):
            PhysicalCamera(provenance={"kind": "measured", "source": "camera"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.json"
            data = {
                "schema": "ecg-camera-calibration-v1",
                "lens": dataclasses.asdict(LensConfig()),
                "provenance": {
                    "kind": "measured",
                    "source": "laboratory fixture",
                    "method": "fitted optical bench",
                    "data_sha256": "a" * 64,
                },
            }
            path.write_text(json.dumps(data))
            a = PhysicalCamera.from_json(path)
            data["lens"]["f_number"] = 4.0
            path.write_text(json.dumps(data))
            b = PhysicalCamera.from_json(path)
            self.assertNotEqual(a.specification(), b.specification())

    def test_display_pwm_matches_piecewise_interval_overlap(self):
        c = DisplayConfig(
            refresh_hz=90.0, duty_cycle=0.27, exposure_s=0.007, phase_cycles=0.19
        )
        display = EmissiveDisplay(c)
        starts = torch.tensor(
            [-0.009, 0.0, 0.001, 0.008, 0.021, 0.033], dtype=torch.float64
        )
        expected = []
        period = 1 / c.refresh_hz
        for start in starts.tolist():
            a, b = (
                start + c.phase_cycles * period,
                start + c.phase_cycles * period + c.exposure_s,
            )
            overlap = sum(
                max(0.0, min(b, (k + c.duty_cycle) * period) - max(a, k * period))
                for k in range(-5, 10)
            )
            expected.append(overlap / c.exposure_s)
        torch.testing.assert_close(
            display.exposure_fraction(starts),
            torch.tensor(expected, dtype=torch.float64),
        )

    def test_continuous_backlight_has_no_refresh_banding(self):
        display = EmissiveDisplay(
            DisplayConfig(duty_cycle=1.0, exposure_s=0.0002, readout_s=0.037)
        )
        torch.testing.assert_close(
            display.exposure_fraction(torch.linspace(0, 0.037, 100)),
            torch.ones(100),
            atol=2e-5,
            rtol=0,
        )

    def test_integral_period_shutter_averages_to_duty_cycle(self):
        display = EmissiveDisplay(
            DisplayConfig(refresh_hz=100.0, duty_cycle=0.35, exposure_s=0.03)
        )
        torch.testing.assert_close(
            display.exposure_fraction(torch.linspace(-0.04, 0.04, 55)),
            torch.full((55,), 0.35),
            atol=1e-6,
            rtol=0,
        )

    def test_display_rolling_shutter_bands_require_pulsed_emission(self):
        field = self.identity_field(96, 128)
        white = torch.ones(1, 3, 96, 128)
        config = DisplayConfig(
            native_width=64,
            native_height=48,
            fill_factor=1.0,
            reflection_strength=0.0,
            footprint_samples=3,
            duty_cycle=0.4,
            exposure_s=0.0007,
        )
        continuous = EmissiveDisplay(dataclasses.replace(config, duty_cycle=1.0))(
            white, field
        )
        pulsed = EmissiveDisplay(config)(white, field)
        self.assertGreater(float(pulsed.mean((0, 1, 3)).std()), 0.15)
        self.assertLess(float(continuous.mean((0, 1, 3)).std()), 2e-5)

    def test_subpixel_order_reverses_red_and_blue_apertures(self):
        field = self.identity_field(64, 192)
        white = torch.ones(1, 3, 64, 192)
        c = DisplayConfig(
            native_width=32,
            native_height=32,
            fill_factor=1.0,
            reflection_strength=0.0,
            footprint_samples=1,
            duty_cycle=1.0,
        )
        rgb = EmissiveDisplay(c)(white, field)
        bgr = EmissiveDisplay(dataclasses.replace(c, subpixel_order="BGR"))(
            white, field
        )
        torch.testing.assert_close(rgb[:, [2, 1, 0]], bgr)
        torch.testing.assert_close(
            rgb.mean(), torch.tensor(0.963276), atol=0.01, rtol=0
        )

    def test_fractional_panel_boundary_has_one_coverage_factor(self):
        display = EmissiveDisplay()
        field = self.identity_field(8, 8)
        background = torch.full((1, 3, 8, 8), 0.2)
        emission = torch.full_like(background, 0.8)
        panel = torch.full((1, 1, 8, 8), 0.5)
        from unittest.mock import patch

        with patch.object(
            display, "frame_coverage", return_value=torch.full_like(panel, 0.75)
        ):
            image, bezel = display.composite(
                background, emission, panel, field, shadow=0.1
            )
        expected = (0.2 * 0.25 + 0.012 * 0.25) * 0.9 + 0.8 * 0.5
        torch.testing.assert_close(image, torch.full_like(image, expected))
        torch.testing.assert_close(bezel, torch.full_like(bezel, 0.25))

    def test_unresolved_display_stripes_average_to_neutral_white(self):
        field = self.identity_field(64, 160)
        c = DisplayConfig(
            native_width=1280,
            native_height=512,
            fill_factor=0.88,
            reflection_strength=0.0,
            footprint_samples=2,
        )
        image = EmissiveDisplay(c)(torch.ones(1, 3, 64, 160), field)
        torch.testing.assert_close(image[:, 0], image[:, 1], atol=2e-4, rtol=0)
        torch.testing.assert_close(image[:, 2], image[:, 1], atol=2e-4, rtol=0)
        self.assertLess(float(image.std()), 1e-4)

    def test_display_uses_rigid_geometry_and_distinct_medium_metadata(self):
        renderer = ECGRenderer(
            self.config(), display=DisplayConfig(native_width=256, native_height=160)
        )
        out = self.render(renderer, return_scene=True)
        self.assertEqual(out.metadata["capture_medium"], "display")
        self.assertTrue(torch.equal(out.reference, self.signal()))
        self.assertTrue(torch.isfinite(out.image).all())
        self.assertGreater(float(out.scene["bezel_mask"].sum()), 0)
        self.assertEqual(float(out.metadata["parameters"]["curl_x"].abs().sum()), 0)
        with self.assertRaises(RuntimeError):
            self.render(renderer, parameters={"curl_x": 0.1})

    def test_template_dimensions_preserve_calibration_and_window_duration(self):
        renderer = ECGRenderer(self.config(), document_template="clinical")
        out = self.render(
            renderer, photo=False, parameters={"layout": "12x1", "speed_index": 1}
        )
        m = out.metadata
        template = DOCUMENT_TEMPLATES["clinical"]
        torch.testing.assert_close(
            m["left"] / m["ppm"], torch.tensor([template.left_mm])
        )
        torch.testing.assert_close(m["top"] / m["ppm"], torch.tensor([template.top_mm]))
        torch.testing.assert_close(
            (m["x1"] - m["x0"]) / (m["ppm"] * m["speed"])[:, None],
            m["end_s"] - m["start_s"],
        )
        self.assertTrue(torch.equal(out.reference, self.signal()))

    def test_all_templates_render_across_layouts(self):
        for template in DOCUMENT_TEMPLATES:
            renderer = ECGRenderer(self.config(), document_template=template)
            for layout in ("12x1", "3x4", "6x2"):
                out = self.render(renderer, photo=False, parameters={"layout": layout})
                self.assertTrue(torch.isfinite(out.image).all())
                self.assertEqual(out.targets["traces"].shape, (1, 12, 144, 192))

    def test_thermal_print_zero_strength_identity_and_generator_isolation(self):
        c = ThermalPrintConfig(
            head_strength=0.0,
            feed_strength=0.0,
            retention_strength=0.0,
            fiber_strength=0.0,
        )
        material = ThermalPrint(64, 96, c)
        generator = torch.Generator().manual_seed(17)
        before = generator.get_state().clone()
        result = material(
            torch.tensor([2.0]), generator=generator, phase=torch.tensor([0.73])
        )
        self.assertTrue(torch.equal(before, generator.get_state()))
        for value in result.values():
            torch.testing.assert_close(value, torch.ones_like(value), atol=0, rtol=0)

    def test_thermal_head_defects_follow_feed_direction(self):
        c = ThermalPrintConfig(
            feed_strength=0.0, retention_strength=0.0, fiber_strength=0.0
        )
        output = ThermalPrint(64, 96, c)(
            torch.tensor([2.0]), phase=torch.tensor([0.37])
        )["print_retention"]
        torch.testing.assert_close(output[..., 0, :], output[..., -1, :])
        self.assertGreater(float(output.std()), 0.01)

    def test_thermal_component_keeps_geometry_random_stream(self):
        config = dataclasses.replace(self.config(), surface_mode="wrinkled")
        a = self.render(ECGRenderer(config))
        b = self.render(ECGRenderer(config, thermal_print=ThermalPrintConfig()))
        self.assertFalse(torch.equal(a.image, b.image))
        self.assertTrue(torch.equal(a.inverse_field, b.inverse_field))
        for key in a.targets:
            self.assertTrue(torch.equal(a.targets[key], b.targets[key]), key)

    def test_scan_physical_scale_and_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.png"
            image = np.zeros((32, 48, 3), dtype=np.uint8)
            image[..., 0] = np.arange(48)[None] * 4
            Image.fromarray(image).save(path)
            bank = MeasuredPaperBank(
                [PaperScan(str(path), 4.0, "test ramp", "CC0", "illustrative")]
            )
            output, ids = bank(torch.tensor([2.0]), 8, 12, phase=torch.tensor([0.0]))
            # Centered 24x16-source-pixel crop, 2 source pixels per output pixel.
            expected = (torch.arange(12) * 2 + 12.5) * (4 / 255)
            torch.testing.assert_close(output[0, 0, 0], expected)
            self.assertEqual(ids.tolist(), [0])
            self.assertEqual(len(bank.specification()["provenance"][0]["sha256"]), 64)
            with self.assertRaises(RuntimeError):
                bank(torch.tensor([0.5]), 16, 16, phase=torch.tensor([0.0]))

    def test_scan_manifest_roundtrip_and_geometry_invariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.png"
            Image.new("RGB", (512, 512), (240, 245, 250)).save(path)
            manifest = Path(tmp) / "paper.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "ecg-paper-scans-v1",
                        "scans": [
                            {
                                "path": "scan.png",
                                "pixels_per_mm": 1.0,
                                "source": "test fixture",
                                "license": "CC0",
                                "kind": "illustrative",
                            }
                        ],
                    }
                )
            )
            bank = MeasuredPaperBank.from_json(manifest)
            a = self.render(ECGRenderer(self.config()))
            b = self.render(ECGRenderer(self.config(), measured_paper=bank))
            self.assertTrue(torch.equal(a.inverse_field, b.inverse_field))
            for key in a.targets:
                self.assertTrue(torch.equal(a.targets[key], b.targets[key]), key)

    def test_extension_configuration_is_checkpoint_identity(self):
        config = self.config()
        a = ECGImageLayer(config=config, thermal_print=ThermalPrintConfig())
        b = ECGImageLayer(
            config=config, thermal_print=ThermalPrintConfig(head_strength=0.3)
        )
        with self.assertRaises(ValueError):
            b.load_state_dict(a.state_dict())
        clone = ECGImageLayer(config=config, thermal_print=ThermalPrintConfig())
        a(self.signal())
        clone.load_state_dict(a.state_dict())
        self.assertTrue(torch.equal(a(self.signal()), clone(self.signal())))

    def test_camera_substitution_uses_depth_contract_without_renderer_type_switch(self):
        class FlatCamera(CameraAppearance):
            def specification(self):
                return {"model": "test-substitution"}

            def scene_depth(self, scene, ppm, width, valid):
                return valid * 0.5

            def forward(self, image, parameters, generator=None, *, depth_m=None):
                if depth_m is None:
                    raise AssertionError("Depth extension contract not called")
                return super().forward(image, parameters, generator, depth_m=depth_m)

        out = self.render(
            ECGRenderer(
                self.config(),
                camera_model=FlatCamera(mode="sensor", jpeg_enabled=False),
            )
        )
        self.assertTrue(torch.isfinite(out.image).all())

    def test_invalid_configurations_fail_at_construction(self):
        for factory in (
            lambda: LensConfig(f_number=True),
            lambda: LensConfig(focus_distance_m=0.001),
            lambda: LensConfig(max_sigma_pixels=5),
            lambda: DisplayConfig(duty_cycle=0),
            lambda: DisplayConfig(native_width=1),
            lambda: DisplayConfig(footprint_samples=True),
            lambda: ThermalPrintConfig(head_strength=float("nan")),
            lambda: DocumentTemplate("invalid", top_mm=2),
            lambda: PaperScan("x", 0, "source", "license"),
            lambda: ECGRenderer(self.config(), camera_model=CameraAppearance()),
            lambda: ECGRenderer(self.config(), document_template="missing"),
            lambda: ECGRenderer(
                dataclasses.replace(self.config(), surface_mode="fold"),
                display=DisplayConfig(),
            ),
            lambda: ECGRenderer(
                self.config(),
                display=DisplayConfig(),
                thermal_print=ThermalPrintConfig(),
            ),
        ):
            with self.assertRaises((ValueError, TypeError)):
                factory()

    def test_sensor_electron_counts_match_poisson_variance_and_read_noise(self):
        sensor = SensorConfig(full_well_electrons=10000.0, read_noise_electrons=3.0)
        camera = PhysicalCamera(sensor=sensor)
        original = {
            "read_noise": torch.tensor([0.1]),
            "shot_noise": torch.tensor([0.2]),
        }
        adjusted = camera.sensor_parameters(original)
        self.assertEqual(float(original["read_noise"][0]), float(torch.tensor(0.1)))
        for radiance in (0.0, 0.1, 1.0):
            expected = (radiance * 10000 + 3**2) / 10000**2
            variance = (
                adjusted["read_noise"].square()
                + adjusted["shot_noise"].square() * radiance
            )
            self.assertAlmostEqual(float(variance), expected, delta=1e-10)

    def test_history_drying_couples_deposit_wetness_and_relief(self):
        p = {"paper_age": torch.zeros(1)}
        wet = MaterialHistory(spill_ml=1.0, minutes_since_spill=0.0).parameters(p)
        dry = MaterialHistory(spill_ml=1.0, minutes_since_spill=10000.0).parameters(p)
        self.assertEqual(float(wet["coffee_strength"][0]), 0.0)
        self.assertGreater(float(dry["coffee_strength"][0]), 0.0)
        self.assertGreater(
            float(wet["water_strength"][0]), float(dry["water_strength"][0])
        )
        self.assertGreater(
            float(wet["history_warp_strength"][0]),
            float(dry["history_warp_strength"][0]),
        )
        self.assertGreater(float(dry["history_warp_strength"][0]), 0.0)
        clean = MaterialHistory().parameters(p)
        self.assertEqual(float(clean["history_warp_strength"][0]), 0.0)

    def test_history_geometry_normals_and_targets_share_the_changed_field(self):
        config = dataclasses.replace(self.config(), surface_mode="flat")
        a = self.render(ECGRenderer(config, material_history=MaterialHistory()))
        b = self.render(
            ECGRenderer(
                config,
                material_history=MaterialHistory(spill_ml=2.0, minutes_since_spill=5.0),
            ),
            return_scene=True,
        )
        self.assertFalse(torch.equal(a.inverse_field, b.inverse_field))
        self.assertGreater(
            float(b.scene["height"].max() - b.scene["height"].min()), 0.001
        )
        self.assertTrue(torch.equal(b.reference, self.signal()))
        rectified = self.render(
            ECGRenderer(config, material_history=MaterialHistory(spill_ml=2.0)),
            photo=False,
        )
        self.assertEqual(
            float(
                rectified.metadata["parameters"]["history_warp_strength"].abs().sum()
            ),
            0.0,
        )

    def test_authorized_scene_manifests_reject_image_and_family_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (32, 32), (100, 120, 80)).save(root / "a.png")
            Image.new("RGB", (32, 32), (90, 110, 80)).save(root / "b.png")
            path = root / "scenes.json"
            first = {
                "path": "a.png",
                "source": "test fixture",
                "license": "CC0",
                "family": "table-a",
                "split": "train",
            }
            second = {**first, "path": "b.png", "family": "table-b", "split": "test"}
            data = {"schema": "ecg-scene-bank-v1", "assets": [first, second]}
            path.write_text(json.dumps(data))
            bank = SceneBank.from_json(path, height=144, width=192)
            renderer = ECGRenderer(self.config(), backgrounds=bank)
            self.assertEqual(
                renderer.specification()["scene_provenance"][0]["family"], "table-a"
            )
            self.assertTrue(torch.isfinite(self.render(renderer).image).all())
            for key, value in (("family", "table-a"), ("path", "a.png")):
                data["assets"][1] = {**second, key: value}
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    SceneBank.from_json(path)

    def test_cli_can_export_template_thermal_camera_and_display(self):
        from ecg_gpu.gpu_cli import main
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "signal.npy", self.signal()[0].numpy())
            (root / "display.json").write_text(
                json.dumps({"native_width": 256, "native_height": 160})
            )
            (root / "lens.json").write_text(
                json.dumps(
                    {
                        "schema": "ecg-camera-calibration-v1",
                        "lens": dataclasses.asdict(LensConfig()),
                        "provenance": {"kind": "illustrative"},
                    }
                )
            )
            common = [
                "ecg-render",
                "--input",
                str(root / "signal.npy"),
                "--fs",
                "500",
                "--width",
                "192",
                "--height",
                "144",
                "--surface",
                "off",
                "--camera-mode",
                "sensor",
                "--no-jpeg",
                "--device",
                "cpu",
            ]
            for name, extra in (
                ("thermal", ["--template", "clinical", "--thermal-print"]),
                ("camera", ["--camera-calibration", str(root / "lens.json")]),
                ("display", ["--display-profile", str(root / "display.json")]),
            ):
                with patch(
                    "sys.argv",
                    common + ["--output", str(root / (name + ".png"))] + extra,
                ):
                    main()
                self.assertTrue((root / (name + ".png")).is_file())

    def test_extensions_train_a_downstream_network_with_finite_gradients(self):
        cases = [
            {
                "thermal_print": ThermalPrintConfig(),
                "camera_model": PhysicalCamera(jpeg_enabled=False),
            },
            {"display": DisplayConfig(native_width=256, native_height=160)},
        ]
        for options in cases:
            layer = ECGImageLayer(config=self.config(), randomize=False, **options)
            model = torch.nn.Sequential(
                layer,
                torch.nn.Conv2d(3, 2, 3, padding=1),
                torch.nn.AdaptiveAvgPool2d(1),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            weight = model[1].weight.detach().clone()
            original = self.signal().clone()
            for _ in range(2):
                optimizer.zero_grad()
                loss = model(original).square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(model[1].weight.grad).all())
                optimizer.step()
            self.assertFalse(torch.equal(weight, model[1].weight))
            self.assertTrue(torch.equal(original, self.signal()))
            self.assertEqual(list(layer.parameters()), [])

    def test_hot_path_has_no_python_cpu_numpy_or_scalar_extraction(self):
        from unittest.mock import patch

        config = dataclasses.replace(self.config(), surface_mode="flat")
        renderer = ECGRenderer(
            config,
            thermal_print=ThermalPrintConfig(),
            camera_model=PhysicalCamera(jpeg_enabled=False),
            material_history=MaterialHistory(spill_ml=1.0),
        )
        with (
            patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("CPU copy in hot path")
            ),
            patch.object(
                torch.Tensor, "numpy", side_effect=AssertionError("NumPy in hot path")
            ),
            patch.object(
                torch.Tensor,
                "item",
                side_effect=AssertionError("Scalar extraction in hot path"),
            ),
        ):
            out = self.render(renderer)
        self.assertTrue(torch.isfinite(out.image).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_new_models_remain_on_cuda(self):
        renderer = ECGRenderer(
            self.config(),
            camera_model=PhysicalCamera(jpeg_enabled=False),
            thermal_print=ThermalPrintConfig(),
            document_template="clinical",
        ).cuda()
        out = renderer(
            self.signal().cuda(),
            500,
            generator=torch.Generator(device="cuda").manual_seed(4),
        )
        self.assertEqual(out.image.device.type, "cuda")
        self.assertEqual(out.reference.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
