"""Prepare a recording/patient-disjoint PM-ECG-ID physical-capture pilot.

Input is the published archive extracted locally. --fetch-missing can retrieve
only the selected members with HTTP ranges, avoiding a 33.9 GB full download.
Each retrieved member is checked against its ZIP CRC, then SHA-256 pinned.
The advertised whole-archive MD5 is NOT verified after a partial download.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import urllib.request
import zipfile
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.signals import LEADS

RECORD_URL = "https://zenodo.org/api/records/13617673"
ARCHIVE_URL = RECORD_URL + "/files/pmcardio_ecg_image_database.zip/content"
ARCHIVE_SIZE = 33863935685
CONDITIONS = ("photos_scans", "photos_samsung", "photos_bents")
LAYOUTS = ("3x4", "3x4+1R", "6x2", "12x1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def get_range(start, end):
    request = urllib.request.Request(
        ARCHIVE_URL, headers={"Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        expected = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
        if response.status != 206 or response.headers.get("Content-Range") != expected:
            raise ValueError("Server did not honor the exact archive byte range")
        raw = response.read(end - start + 2)
    if len(raw) != end - start + 1:
        raise ValueError("Truncated or oversized archive range")
    return raw


class RangeArchive(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Unknown seek mode")
        position = offset + (
            0 if whence == 0 else self.position if whence == 1 else ARCHIVE_SIZE
        )
        if position < 0:
            raise ValueError("Negative archive offset")
        self.position = position
        return position

    def read(self, size=-1):
        size = min(
            ARCHIVE_SIZE - self.position,
            size if size >= 0 else ARCHIVE_SIZE - self.position,
        )
        if size <= 0:
            return b""
        if size > 128 * 1024**2:
            raise ValueError("Unexpected oversized archive read")
        raw = get_range(self.position, self.position + size - 1)
        self.position += size
        return raw


def archive_index(root):
    path = root / "zip-index.json"
    if path.exists():
        return json.loads(path.read_text())
    with zipfile.ZipFile(RangeArchive()) as archive:
        data = [
            {
                "name": x.filename,
                "size": x.file_size,
                "compressed": x.compress_size,
                "offset": x.header_offset,
                "crc": x.CRC,
            }
            for x in archive.infolist()
        ]
    write_json(path, data)
    return data


def fetch_member(root, entry):
    name = entry["name"]
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe archive member name")
    target = root / relative
    if target.exists():
        content = target.read_bytes()
    else:
        offset = entry["offset"]
        header = get_range(offset, offset + 30 + len(name.encode()) + 127)
        fields = struct.unpack("<IHHHHHIIIHH", header[:30])
        signature, _, flags, method = fields[:4]
        name_size, extra_size = fields[-2:]
        if signature != 0x04034B50 or flags & 1 or method not in (0, 8):
            raise ValueError("Unsupported or corrupt ZIP local header")
        if header[30 : 30 + name_size].decode("utf-8") != name:
            raise ValueError("ZIP local filename disagrees with index")
        start = offset + 30 + name_size + extra_size
        payload = get_range(start, start + entry["compressed"] - 1)
        content = zlib.decompress(payload, -15) if method == 8 else payload
        target.parent.mkdir(parents=True, exist_ok=True)
    if len(content) != entry["size"] or zlib.crc32(content) != entry["crc"]:
        raise ValueError("ZIP member length or CRC mismatch: " + name)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(target)
    return target


def excluded_groups(paths):
    patients, recordings = set(), set()
    for path in paths:
        data = json.loads(Path(path).read_text())
        if isinstance(data, dict):
            patients.update(
                "ptbxl-patient-" + str(float(patient))
                for patient in data.get("excluded_patient_ids", [])
            )
        records = data["records"] if isinstance(data, dict) else data
        for record in records:
            patients.add(record["group_id"])
            recordings.add(int(record["source_id"].split("-")[-1]))
    return patients, recordings


def make_reference(short, rhythm, columns):
    short, rhythm = np.asarray(short), np.asarray(rhythm)
    if columns not in (1, 2, 4) or short.shape != (5000 // columns, 12):
        raise ValueError("Unexpected PM-ECG-ID cutout shape")
    # Column order was inspected on original scans. Reference construction
    # does not reach inference or select the model's predicted layout.
    column = (
        [0] * 12
        if columns == 1
        else [0] * 6 + [1] * 6
        if columns == 2
        else [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
    )
    values = np.full((12, 5000), np.nan)
    n = len(short)
    for lead, col in enumerate(column):
        values[lead, col * n : (col + 1) * n] = short[:, lead]
    rhythm_leads = []
    if rhythm.shape[0] != 5000:
        raise ValueError("Expected ten-second PM rhythm signals at 500 Hz")
    for r in range(rhythm.shape[1]):
        matches = [
            lead
            for lead, col in enumerate(column)
            if np.array_equal(rhythm[col * n : (col + 1) * n, r], short[:, lead])
        ]
        if len(matches) != 1:
            raise ValueError("Cannot independently assign PM rhythm lead to short lead")
        lead = matches[0]
        if lead in rhythm_leads:
            raise ValueError("Duplicate rhythm lead")
        values[lead] = rhythm[:, r]
        rhythm_leads.append(lead)
    if (
        np.isinf(values).any()
        or not np.isfinite(short).all()
        or not np.isfinite(rhythm).all()
    ):
        raise ValueError("Invalid source reference samples")
    return values, [LEADS[i] for i in rhythm_leads]


def prepare(root, ptbxl_metadata, exclude, per_layout=2, fetch_missing=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    index = {v["name"]: v for v in archive_index(root)} if fetch_missing else {}
    metadata = root / "metadata.csv"
    if not metadata.exists():
        source = fetch_member(root, index["final_data/metadata.csv"])
        metadata.write_bytes(source.read_bytes())
    rows = list(csv.DictReader(metadata.open()))
    ptb = {int(r["ecg_id"]): r for r in csv.DictReader(Path(ptbxl_metadata).open())}
    excluded_patients, excluded_records = excluded_groups(exclude)
    choices, exclusions, used_patients = [], [], set()
    for layout in LAYOUTS:
        candidates = sorted(
            [
                r
                for r in rows
                if r["ECG format"] == layout
                and r["ECG number of pages"] == "1"
                and r["Image relative path"].startswith("photos_scans/")
            ],
            key=lambda r: hashlib.sha256(
                ("external-pilot-v1/" + r["ECG ID"]).encode()
            ).hexdigest(),
        )
        selected = []
        for row in candidates:
            recording = int(row["ECG ID"].split("_")[1])
            patient = "ptbxl-patient-" + ptb[recording]["patient_id"]
            if (
                recording in excluded_records
                or patient in excluded_patients
                or patient in used_patients
            ):
                exclusions.append(
                    {
                        "recording_id": recording,
                        "patient_id": patient,
                        "reason": "previously used recording/patient or duplicate patient",
                    }
                )
                continue
            selected.append(row)
            used_patients.add(patient)
            if len(selected) == per_layout:
                break
        if len(selected) != per_layout:
            raise ValueError("Insufficient disjoint patients for " + layout)
        choices.extend(selected)
    ids = {r["Image ID"] for r in choices}
    scheduled = sorted(
        [
            r
            for r in rows
            if r["Image ID"] in ids
            and r["Image relative path"].split("/")[0] in CONDITIONS
        ],
        key=lambda r: (
            int(r["Image ID"]),
            CONDITIONS.index(r["Image relative path"].split("/")[0]),
        ),
    )
    if len(scheduled) != len(choices) * len(CONDITIONS):
        raise ValueError("Incomplete capture-condition pairing")
    names = ["final_data/data/leads.npz", "final_data/data/rhythms.npz"]
    names += ["final_data/visual_data/" + r["Image relative path"] for r in scheduled]
    if fetch_missing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda name: fetch_member(root, index[name]), names))
    (root / "references").mkdir(exist_ok=True)
    cases = []
    with (
        np.load(root / names[0], allow_pickle=False) as leads,
        np.load(root / names[1], allow_pickle=False) as rhythms,
    ):
        for row in scheduled:
            source_id = row["ECG ID"]
            reference, rhythm_names = make_reference(
                leads[source_id],
                rhythms[source_id],
                int(row["ECG number of columns per page"]),
            )
            ref = root / "references" / f"{source_id}.npz"
            if not ref.exists():
                np.savez_compressed(
                    ref,
                    values_mv=reference,
                    fs=np.asarray(500),
                    lead_names=np.asarray(LEADS),
                )
            image = root / "final_data/visual_data" / row["Image relative path"]
            recording = int(source_id.split("_")[1])
            condition = row["Image relative path"].split("/")[0]
            cases.append(
                {
                    "image_id": f"pm-{row['Image ID']}-{condition}",
                    "patient_id": "ptbxl-patient-" + ptb[recording]["patient_id"],
                    "recording_id": recording,
                    "condition": condition,
                    "image": str(image.relative_to(root)),
                    "image_sha256": digest(image),
                    "reference": str(ref.relative_to(root)),
                    "reference_sha256": digest(ref),
                    "reference_layout_scoring_only": row["ECG format"],
                    "reference_rhythm_leads_scoring_only": rhythm_names,
                    "reference_printed_samples": int(np.isfinite(reference).sum()),
                }
            )
    manifest = {
        "schema": "pm-ecg-external-pilot-v1",
        "source": "https://zenodo.org/records/13617673",
        "doi": "10.5281/zenodo.13617673",
        "source_license": "GPL-3.0-or-later",
        "status": "frozen_before_inference",
        "selection": "Two patients per layout, SHA256 external-pilot-v1/ECG-ID order",
        "conditions": list(CONDITIONS),
        "layouts": list(LAYOUTS),
        "patients": len(used_patients),
        "images": len(cases),
        "source_metadata_rows": len(rows),
        "source_metadata_sha256": digest(metadata),
        "ptbxl_metadata_sha256": digest(ptbxl_metadata),
        "excluded_manifest_sha256": {Path(p).name: digest(p) for p in exclude},
        "exclusions_before_inference": exclusions,
        "our_prior_patient_overlap": False,
        "upstream_training_overlap": "unknown",
        "reference_fs": 500,
        "reference_units": "mV",
        "reference_time": "Printed document horizontal time axis, columns retained. Long rhythm identity checked by exact source samples.",
        "source_arrays_sha256": {Path(n).name: digest(root / n) for n in names[:2]},
        "whole_archive_md5_verified": False,
        "partial_download_crc_checked": fetch_missing,
        "cases": cases,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ptbxl-metadata", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--per-layout", type=int, default=2)
    parser.add_argument("--fetch-missing", action="store_true")
    args = parser.parse_args()
    if args.per_layout < 1:
        parser.error("per-layout must be positive")
    result = prepare(**vars(args))
    print(
        json.dumps(
            {
                "images": result["images"],
                "patients": result["patients"],
                "source": result["source"],
            }
        )
    )
