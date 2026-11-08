"""Analytical occlusion control on the unchanged native normalization method.

This is not full model inference or an image benchmark. A fixed physical ECG
peak stays at four seconds while only leading observations become missing.
The control measures how native extent-based interpolation changes that time.
"""

import argparse
import json
from pathlib import Path
import sys

from reproductions.common import sha256, verify_source, write_json
from reproductions.npj.protocol import COMMIT


def measure(upstream):
    source = verify_source(upstream, COMMIT)
    root = Path(upstream).resolve()
    sys.path.insert(0, str(root))
    import torch
    from src.model.lead_identifier import LeadIdentifier
    identifier = LeadIdentifier.__new__(LeadIdentifier)
    identifier.required_valid_samples = 3
    identifier.target_num_samples = 5000
    fs, ppmm, speed = 500, 20, 25
    t = torch.arange(5000, dtype=torch.float64) / fs
    peak = torch.exp(-((t - 4.) / .02) ** 2)
    raw = torch.stack([1000 + 200 * k - 100 * peak for k in range(3)])
    rows = []
    for missing in [0, 100, 200, 500]:
        observed = raw.clone()
        observed[:, :missing] = torch.nan
        output = identifier.normalize(observed, avg_pixel_per_mm=ppmm, mv_per_mm=.1)
        peak_index = int(output[0].argmin())
        rows.append({"leading_missing_columns": missing, "physical_missing_duration_s": missing / (ppmm * speed),
                     "true_peak_s": 4., "native_peak_s": peak_index / fs,
                     "native_peak_error_ms": (peak_index / fs - 4) * 1000,
                     "input_missing_samples": int(torch.isnan(observed).sum()),
                     "native_output_missing_samples": int(torch.isnan(output).sum())})
    return {"schema": "native-normalization-time-axis-control-v1", "scope": "unchanged native normalization, no neural network or image inference",
        "source": source, "native_module_sha256": sha256(root / "src/model/lead_identifier.py"),
        "fs": fs, "pixels_per_mm": ppmm, "assumed_speed_mm_per_s": speed, "output_samples": 5000,
        "physical_signal_changed": False, "only_leading_visibility_changed": True, "rows": rows,
        "real_photo_error_cause_proven": False,
        "limitation": "This proves the mechanism on a specified analytical fixture, not that it explains every real-image delay"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = measure(args.upstream)
    write_json(args.out, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
