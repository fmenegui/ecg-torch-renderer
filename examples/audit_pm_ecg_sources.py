"""Check PM cutout timing/identity against original hash-recorded PTB-XL WFDB.

PM-ECG-ID shifts the source DC level. This audit measures that constant offset
and verifies residual waveform agreement. It does not alter scoring references
or fit/amend any inference output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.data import read_wfdb16


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, original_wfdb, out):
    root, original_wfdb = Path(root), Path(original_wfdb)
    manifest = json.loads((root / "manifest.json").read_text())
    selected = {r["recording_id"]: r for r in manifest["cases"]}
    names = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    checks = []
    for recording, row in sorted(selected.items()):
        header = original_wfdb / f"{recording:05d}_hr.hea"
        original = read_wfdb16(header)
        with np.load(root / row["reference"], allow_pickle=False) as z:
            reference = z["values_mv"]
        for lead, name in enumerate(names):
            mask = np.isfinite(reference[lead])
            full = original[name].values
            if original[name].fs != 500 or len(full) != 5000:
                raise ValueError("Wrong source duration or sample rate")
            difference = reference[lead, mask] - full[mask]
            offset = float(np.mean(difference))
            residual = float(np.max(np.abs(difference - offset)))
            if residual > 1e-6:
                raise ValueError(
                    "Source waveform/time/lead mismatch: " + str(recording) + " " + name
                )
            checks.append(
                {
                    "recording_id": recording,
                    "lead": name,
                    "samples": int(mask.sum()),
                    "pm_source_dc_offset_mv": offset,
                    "max_residual_after_constant_dc_mv": residual,
                    "header_sha256": digest(header),
                    "data_sha256": digest(header.with_suffix(".dat")),
                }
            )
    result = {
        "schema": "external-m7-original-source-audit-v1",
        "passed": True,
        "records": len(selected),
        "lead_checks": len(checks),
        "source": "https://physionet.org/content/ptb-xl/1.0.3/",
        "source_license": "CC-BY-4.0",
        "WFDB_initial_values_and_checksums_verified": True,
        "max_centered_residual_mv": max(
            r["max_residual_after_constant_dc_mv"] for r in checks
        ),
        "scoring_references_modified": False,
        "note": "Constant source DC offsets are reported, not removed from raw scores. No method output is used by this data audit.",
        "checks": checks,
    }
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "original-wfdb", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    audit(**vars(parser.parse_args()))
