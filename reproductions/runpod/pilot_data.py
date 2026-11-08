"""Version-2 public pilot inputs. Gold files are separate from image-only inputs."""

import io
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

from reproductions.common import sha256, write_json
from reproductions.manifest import LEADS

FILES = {
    "D0/D0_000022/D0_000022.dat": 120000,
    "D0/D0_000022/D0_000022.hea": 631,
    "D0/D0_000022/D0_000022_02.png": 372153,
    "D2/D2_000001/D2_000001.dat": 120000,
    "D2/D2_000001/D2_000001.hea": 601,
    "D2/D2_000001/D2_000001_l00.jpg": 4218120,
    "D2/D2_000001/D2_000001_l01.jpg": 2647518,
    "D2/D2_000001/D2_000001_l02.jpg": 2213644,
    "D2/D2_000001/D2_000001_p00.jpg": 4457577,
    "D2/D2_000001/D2_000001_p01.jpg": 2893980,
    "D2/D2_000001/D2_000001_p02.jpg": 2352044,
}


def fetch(root, include_ahus=True):
    root = Path(root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, size in FILES.items():
        if not include_ahus and source.startswith("D2/"):
            continue
        path = raw / Path(source).name
        if not path.exists():
            url = "https://www.kaggle.com/api/v1/datasets/download/physionet/ecg-image-database/" + urllib.parse.quote(source, safe="") + "?datasetVersionNumber=2"
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read(size + 1)
            if zipfile.is_zipfile(io.BytesIO(payload)):
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    members = [i for i in archive.infolist() if Path(i.filename).name == path.name and i.file_size == size]
                    if len(members) != 1:
                        raise ValueError("Ambiguous public file download")
                    payload = archive.read(members[0])
            if len(payload) != size:
                raise ValueError("Public file size differs from the complete version-2 inventory")
            path.write_bytes(payload)
        if path.stat().st_size != size:
            raise ValueError("Changed cached pilot object")
        rows.append({"source": source, "bytes": size, "sha256": sha256(path)})
    write_json(root / "acquisition.json", {"schema": "public-pilot-acquisition-v1", "dataset": "physionet/ecg-image-database", "version": 2, "files": rows, "digests": "first-observed and retained, original publisher SHA256 not supplied", "cohort": "two source records, development pilot, not an exact article cohort"})
    return raw


def prepare(root, include_ahus=True):
    import numpy as np
    import pandas as pd
    # WFDB 4.3.0 builds annotation tables with ndarray string indexers at import.
    # Keep its import compatible with pandas 3 without changing signal reading.
    with pd.option_context("future.infer_string", False):
        import wfdb
    root = Path(root)
    raw = fetch(root, include_ahus)
    (root / "gold").mkdir(exist_ok=True)
    groups = {"d0": [], "limb": [], "precordial": []}
    for record in ["D0_000022", "D2_000001"]:
        if not include_ahus and record.startswith("D2"):
            continue
        reference = wfdb.rdrecord(str(raw / record))
        order = [reference.sig_name.index(next(n for n in reference.sig_name if n.upper() == lead.upper())) for lead in LEADS]
        if any(reference.units[i] != "mV" for i in order) or reference.sig_len != 5000:
            raise ValueError("Unexpected WFDB reference units or length")
        values = reference.p_signal[:, order].T
        for image in sorted(raw.glob(record + "_*")):
            if image.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            mask = np.zeros_like(values, dtype=bool)
            if record.startswith("D0"):
                family = "d0"
                layout = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
                for names in layout:
                    for col, lead in enumerate(names):
                        mask[LEADS.index(lead), col * 1250:(col + 1) * 1250] = True
                mask[1] = True
            else:
                family = "limb" if image.stem.rsplit("_", 1)[1].startswith("l") else "precordial"
                mask[:6 if family == "limb" else 0] = family == "limb"
                if family == "precordial":
                    mask[6:] = True
            if not np.isfinite(values[mask]).all():
                raise ValueError("Reference has missing samples inside printed support")
            gold = root / "gold" / (image.stem + ".npz")
            np.savez_compressed(gold, values=values, printed=mask)
            case = {"case_id": image.stem, "patient_id": "source-record:" + record, "split": "development", "image": str(image.relative_to(root)), "image_sha256": sha256(image), "reference": str(gold.relative_to(root)), "reference_sha256": sha256(gold), "reference_unit": "mV", "fs": reference.fs, "acquisition": image.stem.rsplit("_", 1)[1], "page_family": family}
            groups[family].append(case)
    for family, cases in groups.items():
        write_json(root / (family + ".json"), {"schema": "ecg-reproduction-manifest-v1", "cohort_identity_verified": False, "case_grouping": "source record, not asserted biological identity across records", "cases": cases})
    competition = root / "kaggle-image-only"
    (competition / "test").mkdir(parents=True, exist_ok=True)
    case = groups["d0"][0]
    (competition / "test" / (case["case_id"] + ".png")).write_bytes((root / case["image"]).read_bytes())
    pd.DataFrame([{"id": case["case_id"], "lead": lead, "fs": 500, "number_of_rows": 5000 if lead == "II" else 1250} for lead in LEADS]).to_csv(competition / "test.csv", index=False)
    return groups
