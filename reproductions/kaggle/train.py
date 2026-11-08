"""Launch the selected author's original training entrypoint with pinned inputs.

The six-model submission requires six separate training runs. The public B0
default and the selected B7 architecture are different, recorded profiles.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from reproductions.common import environment, new_run, require_cuda, run_bounded, sha256, verify_source, write_json

COMMIT = "9a7ce35b6cb80529c3555efdee434bbbf76a9baf"
PROFILES = {
    "whole_b7": ("whole_model.yaml", "tu-timm/tf_efficientnet_b7.ns_jft_in1k", None, 0),
    "whole_v2_l": ("whole_model.yaml", "tu-timm/tf_efficientnetv2_l.in21k", None, None),
    "series_b6_synthetic": ("series_model.yaml", "tu-timm/tf_efficientnet_b6.ns_jft_in1k", "shared_conv2d", None),
    "series_b6": ("series_model.yaml", "tu-timm/tf_efficientnet_b6.ns_jft_in1k", "shared_conv2d", 0),
    "series_v2_l_3d": ("series_model.yaml", "tu-timm/tf_efficientnetv2_l.in21k", "conv3d", 0),
    "series_v2_l_2d": ("series_model.yaml", "tu-timm/tf_efficientnetv2_l.in21k", "conv2d", 0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--data-contract", required=True, help="Explicit source paths and synthesis recipe")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    require_cuda("cuda")
    upstream = Path(args.upstream).resolve()
    source = verify_source(upstream, COMMIT)
    contract = json.loads(Path(args.data_contract).read_text())
    for field in ["csv_path", "mask_dir", "ecg_dir", "csv_sha256", "num_synthesis_data", "split_provenance"]:
        if field not in contract:
            raise ValueError(f"Missing data contract field: {field}")
    if sha256(contract["csv_path"]) != contract["csv_sha256"]:
        raise ValueError("Changed training cohort/folds")
    config_file, encoder, fusion, synthesis = PROFILES[args.profile]
    if synthesis is None:
        synthesis = contract["num_synthesis_data"]
        if synthesis <= 0:
            raise ValueError("The synthetic-data model requires an explicit nonzero synthesis recipe")
    out = new_run(args.out)
    overrides = [f"model.encoder_name={encoder}", f"cv.val_fold={args.fold}", "wandb.enabled=false", f"data.num_synthesis_data={synthesis}", f"data.csv_path={Path(contract['csv_path']).resolve()}", f"data.mask_dir={Path(contract['mask_dir']).resolve()}", f"data.ecg_dir={Path(contract['ecg_dir']).resolve()}"]
    if fusion:
        overrides.append(f"model.fusion_type={fusion}")
    command = [sys.executable, "train.py", "--config", "configs/" + config_file, "--output_dir", str(out / "checkpoints"), *overrides]
    receipt = {"schema": "kaggle-native-training-v1", "status": "started", "source": source, "profile": args.profile, "environment": environment(), "data_contract_sha256": sha256(args.data_contract), "command": command, "author_refactor_parity_with_final_notebook": "unverified", "reproduction_of_hidden_leaderboard": False}
    write_json(out / "receipt.json", receipt)
    try:
        with (out / "train.log").open("w") as log:
            code = run_bounded(command, args.timeout_seconds, log, cwd=upstream)
        receipt.update(status="completed" if code == 0 else "failed", returncode=code)
    except subprocess.TimeoutExpired:
        receipt["status"] = "timeout"
    finally:
        write_json(out / "receipt.json", receipt)
    if receipt["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
