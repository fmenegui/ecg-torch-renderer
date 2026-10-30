"""Reproduce source/configuration discrepancies and native scorer controls."""

import argparse
import importlib.util
from pathlib import Path
import sys
import warnings

import numpy as np

from reproductions.common import environment, sha256, verify_source, write_json
from reproductions.npj.protocol import COMMIT, WEIGHTS, training_config


def audit(upstream):
    upstream = Path(upstream).resolve()
    identity = verify_source(upstream, COMMIT)
    sys.path.insert(0, str(upstream))
    import yaml
    native = yaml.safe_load((upstream / "src/config/unet.yml").read_text())
    differences = []
    for name, actual, expected in [
        ("epochs", native["TRAIN"]["KWARGS"]["epochs"], 45),
        ("weight_decay", native["TRAIN"]["OPTIMIZER"]["KWARGS"]["weight_decay"], 0.001),
        ("cosine_batches", native["TRAIN"]["LR_SCHEDULER"]["KWARGS"]["T_max"], 20000),
    ]:
        differences.append({"parameter": name, "public_yaml": actual, "article_table4": expected, "different": actual != expected})
    spec = importlib.util.spec_from_file_location("native_npj_evaluate", upstream / "src/evaluate.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    t = np.arange(1000, dtype=float) / 1000
    signal = (np.sin(2 * np.pi * 7.37 * t) + 0.17 * np.sin(2 * np.pi * 13.1 * t))[:, None]
    controls = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, prediction in [
            ("perfect", signal.copy()),
            ("all_missing", np.full_like(signal, np.nan)),
            ("zero", np.zeros_like(signal)),
            ("small_noise", signal + 0.001 * np.cos(2 * np.pi * 23 * t)[:, None]),
        ]:
            result = evaluator.compute_metrics(signal.copy(), prediction, n_translations=0)
            controls[name] = {k: [float(x) for x in v] for k, v in result.items()}
    hashes = {name: sha256(upstream / "weights" / name) for name in WEIGHTS}
    if hashes != WEIGHTS:
        raise ValueError("Published weight identity mismatch")
    paper = training_config(upstream, "paper_table4", "/example/train", "/example/val")
    return {
        "schema": "npj-source-audit-v1", "source": identity, "environment": environment(),
        "weight_sha256": hashes, "training_discrepancies": differences,
        "paper_profile_epochs": paper["TRAIN"]["KWARGS"]["epochs"],
        "native_evaluator_controls": controls,
        "interpretation": [
            "Exact perfect reconstruction has zero noise and receives NaN SNR in this pinned native scorer.",
            "All-missing reconstruction retains the initial nans_fraction=0.0 and undefined errors.",
            "Native main iterates discovered prediction CSVs, so a missing output file has no scheduled row.",
            "These finite controls establish implementation behavior, not an explanation of the published cohort scores.",
        ],
        "training_completed": False, "paper_cohort_reproduction_completed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_json(args.out, audit(args.upstream))


if __name__ == "__main__":
    main()
