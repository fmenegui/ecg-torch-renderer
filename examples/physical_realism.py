"""Export actual renderer outputs and reproducible CPU/GPU timing evidence.

python examples/physical_realism.py --out build/realism --pathtrace --benchmark
No real photographs, patient records or pretrained models are downloaded.
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ecg_gpu import (
    ECGImageLayer,
    RenderConfig,
    ThermalPrintConfig,
    MaterialHistory,
    PhysicalCamera,
    LensConfig,
    SensorConfig,
    DisplayConfig,
)
from ecg_gpu.demo import demo_signals


def export(
    out, *, width=640, height=480, device="cpu", pathtrace=False, benchmark=False
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    values, fs, _ = demo_signals(fs=500, duration=10)
    signals = torch.tensor(values, device=device)[None]
    config = RenderConfig(
        width=width,
        height=height,
        profile="photo",
        severity=0.35,
        surface_mode="fold",
        camera_mode="sensor",
        jpeg_enabled=False,
        jpeg_probability=0,
    )
    parameters = {
        "layout": "3x4+r1",
        "speed_index": 1,
        "gain_index": 1,
        "quarter": 0,
        "angle": -0.08,
        "zoom": 0.83,
        "translate_x": 0.0,
        "translate_y": 0.0,
        "perspective_x": 0.04,
        "perspective_y": -0.03,
        "occlusion_active": 0.0,
        "pen_active": 0.0,
        "artifact_present": [0.0, 0.0, 0.0, 0.0],
        "tear_active": 0.0,
        "hole_active": [0.0, 0.0, 0.0],
        "cut_active": [0.0, 0.0, 0.0],
        "missing_corner_active": 0.0,
        "edge_wear_strength": 0.0,
        "object_shadow": 0,
        "read_noise": 0.001,
        "shot_noise": 0.002,
        "blur": 0.25,
        "motion": 0.0,
        "grid_style": 0,
        "exposure": 1.0,
        "light_x": 0.0,
        "light_y": 0.0,
        "white_balance": [1.0, 1.0, 1.0],
        "gamma": 1.0,
    }
    cases = [
        ("legacy", config, {}),
        (
            "clinical_thermal",
            config,
            {"document_template": "clinical", "thermal_print": ThermalPrintConfig()},
        ),
        (
            "archive_history",
            config,
            {
                "document_template": "archive",
                "thermal_print": ThermalPrintConfig(),
                "material_history": MaterialHistory(
                    spill_ml=1.5,
                    minutes_since_spill=40.0,
                    storage_days=300.0,
                    uv_dose_kj_m2=25.0,
                ),
            },
        ),
        (
            "lens_depth",
            config,
            {
                "document_template": "clinical",
                "camera_model": PhysicalCamera(
                    LensConfig(focus_distance_m=0.18),
                    jpeg_enabled=False,
                    sensor=SensorConfig(),
                ),
            },
        ),
        (
            "display_continuous",
            dataclasses.replace(config, surface_mode="off"),
            {
                "document_template": "compact",
                "display": DisplayConfig(
                    native_width=1024, native_height=768, footprint_samples=4
                ),
            },
        ),
        (
            "display_pwm",
            dataclasses.replace(config, surface_mode="off"),
            {
                "document_template": "compact",
                "display": DisplayConfig(
                    native_width=1024,
                    native_height=768,
                    footprint_samples=4,
                    duty_cycle=0.55,
                    # A typical indoor phone exposure gives partial banding. Very
                    # short exposures (about 1 ms) give fully dark bands.
                    exposure_s=0.004,
                ),
            },
        ),
    ]
    reports, images = [], []
    for name, current, options in cases:
        layer = (
            ECGImageLayer(
                fs,
                config=current,
                parameters=parameters,
                seed=20261020,
                randomize=False,
                **options,
            )
            .to(device)
            .eval()
        )
        batch = layer.render_batch(
            signals, stage="all", return_instances=True, return_field=True
        )
        array = (
            (batch.image[0].cpu().permute(1, 2, 0).numpy() * 255)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
        image = Image.fromarray(array)
        image.save(out / (name + ".png"))
        images.append((name, image))
        report = {
            "name": name,
            "seed": 20261020,
            "specification": layer.renderer.specification(),
            "image_sha256": hashlib.sha256(
                (out / (name + ".png")).read_bytes()
            ).hexdigest(),
            "source_preserved": torch.equal(batch.reference, signals),
            "valid_fraction": float(batch.targets["valid"].mean()),
            "timing": None,
        }
        if benchmark:
            for _ in range(2):
                layer(signals)
            times = []
            for _ in range(7):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                layer(signals)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            report["timing"] = {
                "median_ms": statistics.median(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "repeats": len(times),
                "stage": "image",
                "batch_size": 1,
                "warmup": 2,
            }
        reports.append(report)
    if pathtrace:
        from ecg_gpu.pathtrace import (
            PathTraceConfig,
            pathtrace_capture,
            capture_geometry_assets,
        )
        from ecg_gpu.paper_mechanics import SheetConfig

        layer = ECGImageLayer(
            fs,
            config=dataclasses.replace(config, profile="clean"),
            parameters={"layout": "3x4+r1", "speed_index": 1},
            randomize=False,
            document_template="clinical",
            photo=False,
        ).eval()
        canonical = layer.render_batch(
            signals.cpu(), stage="all", return_instances=True, as_texture=True
        )
        capture = pathtrace_capture(
            canonical,
            config=PathTraceConfig(
                width=width,
                height=height,
                samples_per_pixel=64,
                mesh_columns=33,
                mesh_rows=25,
                surface="multifold",
                edge_tear=0.12,
                hole_radius=0.07,
                sheet=SheetConfig(iterations=72),
                paper_thickness_m=0.00006,
                edge_fibers=32,
                cut_lip_lift_m=0.00015,
                occluder=True,
            ),
            return_geometry=True,
        )
        bank = capture_geometry_assets(capture)
        bank.to_npz(out / "geometry-bank.npz")
        array = (
            (capture["image"][0].permute(1, 2, 0).numpy() * 255)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
        image = Image.fromarray(array)
        image.save(out / "offline_sheet.png")
        images.append(("offline_sheet", image))
        reports.append(
            {
                "name": "offline_sheet",
                "metadata": capture["metadata"],
                "image_sha256": hashlib.sha256(
                    (out / "offline_sheet.png").read_bytes()
                ).hexdigest(),
                "bank_sha256": hashlib.sha256(
                    (out / "geometry-bank.npz").read_bytes()
                ).hexdigest(),
            }
        )
    tile_w, tile_h = width, height
    rows = (len(images) + 1) // 2
    sheet = Image.new("RGB", (tile_w * 2, rows * (tile_h + 36) + 48), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    for i, (name, image) in enumerate(images):
        x, y = i % 2 * tile_w, i // 2 * (tile_h + 36)
        draw.text((x + 10, y + 8), name, fill="#252525", font=font)
        sheet.paste(image, (x, y + 36))
    draw.text(
        (10, rows * (tile_h + 36) + 12),
        "Artificial ECG. Actual renderer output. External realism validation pending.",
        fill="#252525",
        font=font,
    )
    sheet.save(out / "gallery.jpg", quality=94, subsampling=0)
    report = {
        "schema": "ecg-realism-evidence-v1",
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "threads": torch.get_num_threads(),
        "size": [height, width],
        "reference_sha256": hashlib.sha256(
            np.asarray(values, dtype="<f4").tobytes()
        ).hexdigest(),
        "real_image_validation": False,
        "cases": reports,
    }
    (out / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--pathtrace", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = export(
        args.out,
        width=args.width,
        height=args.height,
        device=args.device,
        pathtrace=args.pathtrace,
        benchmark=args.benchmark,
    )
    print(
        json.dumps(
            {
                "cases": len(result["cases"]),
                "device": result["device"],
                "out": str(args.out),
            }
        )
    )
