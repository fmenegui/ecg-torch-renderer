"""Fail closed on ambiguous units, mutable inputs and patient leakage."""

from __future__ import annotations

import json
import math
from pathlib import Path

from reproductions.common import sha256

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
UNIT_TO_MV = {"mV": 1.0, "uV": 0.001, "V": 1000.0}


def validate_manifest(manifest: dict, root: str | Path, verify_bytes: bool = True) -> list[dict]:
    root = Path(root)
    if manifest.get("schema") != "ecg-reproduction-manifest-v1":
        raise ValueError("Unsupported manifest schema")
    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError("An empty cohort is not an evaluation")
    seen = set()
    ownership = {}
    for case in cases:
        for key in ["case_id", "patient_id", "split", "image", "image_sha256", "reference", "reference_sha256", "reference_unit", "fs", "acquisition"]:
            if key not in case:
                raise ValueError(f"Missing manifest field: {key}")
        if case["case_id"] in seen:
            raise ValueError("Duplicate case ID")
        seen.add(case["case_id"])
        if not isinstance(case["patient_id"], str) or not case["patient_id"].strip():
            raise ValueError("patient_id must be an explicit, nonempty source identity")
        patient = case["patient_id"]
        if case["split"] not in ["train", "validation", "test", "development"]:
            raise ValueError("Unknown split")
        previous = ownership.setdefault(patient, case["split"])
        if previous != case["split"]:
            raise ValueError("Patient occurs in multiple splits")
        if case["reference_unit"] not in UNIT_TO_MV:
            raise ValueError("Explicit reference units are required")
        if not isinstance(case["fs"], (int, float)) or not math.isfinite(case["fs"]) or case["fs"] <= 0:
            raise ValueError("Invalid reference sampling rate")
        if verify_bytes:
            for key in ["image", "reference"]:
                if sha256(root / case[key]) != case[key + "_sha256"]:
                    raise ValueError(f"Changed {key} bytes for {case['case_id']}")
    return cases


def load_manifest(path: str | Path) -> tuple[dict, list[dict]]:
    path = Path(path)
    manifest = json.loads(path.read_text())
    return manifest, validate_manifest(manifest, path.parent)


def inference_case(case: dict) -> dict:
    """A narrow handoff: gold layout, reference and ownership cannot leak."""
    result = {k: case[k] for k in ["case_id", "image", "image_sha256"]}
    if "public_metadata" in case:
        allowed = {"duration_seconds", "output_fs", "gain_mm_per_mv", "speed_mm_per_s"}
        if set(case["public_metadata"]) - allowed:
            raise ValueError("Unknown public inference metadata")
        result["public_metadata"] = case["public_metadata"]
    return result
