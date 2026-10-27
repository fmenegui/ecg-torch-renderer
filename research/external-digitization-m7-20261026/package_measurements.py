"""Package frozen waveform/geometry measurements without redistributing photos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(dataset, run, results, out):
    dataset, run, results, out = map(Path, (dataset, run, results, out))
    report = json.loads((run / "report.json").read_text())
    source_manifest = json.loads((dataset / "manifest.json").read_text())
    audit = json.loads((results / "independent-audit.json").read_text())
    if report["status"] != "completed" or not audit["passed"]:
        raise ValueError("Completed independently audited results required")
    files = []
    for path in sorted(run.rglob("*")):
        if (
            path.is_file()
            and path.suffix in (".npz", ".json", ".log", ".txt", ".tsv")
            and not path.name.startswith(".")
        ):
            files.append((path, "run/" + str(path.relative_to(run))))
    for directory in ("references", "original-wfdb"):
        for path in sorted((dataset / directory).glob("*")):
            if path.is_file():
                files.append((path, "inputs/" + directory + "/" + path.name))
    files += [(dataset / "manifest.json", "inputs/manifest.json")]
    for path in sorted(results.glob("*")):
        if path.is_file() and path.suffix in (
            ".json",
            ".csv",
            ".md",
            ".py",
            ".svg",
            ".png",
        ):
            files.append((path, "summary/" + path.name))
    for name in ("run.log", "full-unittest.log", "prepare-final.log"):
        if (dataset / name).exists():
            files.append((dataset / name, "execution/" + name))
    provenance = {
        "schema": "external-m7-measurement-package-v1",
        "images": source_manifest["images"],
        "patients": source_manifest["patients"],
        "case_method_results": len(report["cases"]),
        "original_photos_included": False,
        "photo_replay": "Download the 24 SHA256-pinned original JPEGs from PM-ECG-ID with prepare_pm_ecg_external.py. Paths in inputs/manifest.json are relative to the acquired dataset root.",
        "reference_license": "PM-ECG-ID, Iring et al., DOI 10.5281/zenodo.13617673, GPL-3.0-or-later",
        "original_WFDB_license": "PTB-XL 1.0.3, CC-BY-4.0",
        "upstream": "Open-ECG-Digitizer, Stenhede et al., DOI 10.1038/s41746-025-02327-1, CC-BY-SA-4.0. Code and weights supplied separately.",
        "files": [],
    }
    with zipfile.ZipFile(
        out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path, name in files:
            content = path.read_bytes()
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
            provenance["files"].append(
                {
                    "path": name,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        archive.writestr("manifest.json", json.dumps(provenance, indent=2) + "\n")
    with zipfile.ZipFile(out) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC failed")
        for item in provenance["files"]:
            if hashlib.sha256(archive.read(item["path"])).hexdigest() != item["sha256"]:
                raise ValueError("Packaged measurement hash mismatch")
    receipt = {
        "file": out.name,
        "bytes": out.stat().st_size,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "payload_files": len(files),
        "all_member_CRC_and_SHA256_verified": True,
    }
    (results / "measurement-archive.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "run", "results", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    package(**vars(parser.parse_args()))
