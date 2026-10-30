"""Reference-free decision replay on one already inspected GPU development case.

These outputs helped calibrate v2. This is a regression, not new held-out
evidence and not new GPU inference. No reference waveform is opened here.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecg_gpu.digitization.selective import SelectionPolicy, SignalProposal, SpecialistEvidence, select_proposal
from reproductions.common import sha256, write_json
from reproductions.evaluate import kaggle_timeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-outputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = args.pilot_outputs
    native = json.loads((root / "npj-d0-500hz/receipt.json").read_text())
    kaggle = json.loads((root / "kaggle-d0/receipt.json").read_text())
    record = native["results"][0]
    identity = record["image_sha256"]
    if kaggle["images_sha256"].get(record["case_id"]) != identity:
        raise ValueError("The native outputs came from different source images")
    a = root / "npj-d0-500hz/D0_000022_02.npz"
    b = root / "kaggle-d0/native/digitalised/D0_000022_02.series.npy"
    if sha256(a) != record["output_sha256"] or native["output_unit"] != "uV":
        raise ValueError("Native npj identity or unit changed")
    with np.load(a, allow_pickle=False) as z:
        direct = SignalProposal(z["values"].astype(float) / 1000, 500, identity, record["predicted_layout"])
    candidate = SignalProposal(kaggle_timeline(np.load(b, allow_pickle=False)), 500, identity, "3x4+1R")
    evidence = SpecialistEvidence(identity, "kaggle-second-final-six", kaggle["assets_lock_sha256"])
    selected, decision = select_proposal(direct, candidate, evidence)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "selected.npz", values_mv=selected.values_mv, fs=selected.fs)
    write_json(args.out / "decision.json", {"policy": asdict(SelectionPolicy()), "decision": decision,
        "source_image_sha256": identity, "input_npz_sha256": sha256(a), "native_series_sha256": sha256(b),
        "selected_npz_sha256": sha256(args.out / "selected.npz"), "new_inference": False,
        "policy_calibrated_on_this_known_case": True, "held_out": False, "reference_waveform_opened": False})
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
