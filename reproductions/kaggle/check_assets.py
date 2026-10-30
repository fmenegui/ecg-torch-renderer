"""Strict native checkpoint/architecture loading, without a GPU benchmark claim."""

import argparse
import gc
import importlib
import json
from pathlib import Path
import sys

from reproductions.common import environment, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    root = Path(args.assets).resolve()
    lock = json.loads((root / "assets.lock.json").read_text())
    for name, digest in lock["files"].items():
        if sha256(root / name) != digest:
            raise ValueError("Asset digest mismatch")
    sys.path.insert(0, str(root / "heng"))
    sys.path.insert(0, str(root / "second"))
    helpers = []
    for module in ["stage0_common", "stage1_common", "stage2_common"]:
        importlib.import_module(module)
        helpers.append({"module": module, "status": "imported_original_helper"})
    records = []
    targets = [
        ("heng/weight/stage0-last.checkpoint.pth", "stage0_model", {"pretrained": False}),
        ("heng/weight/stage1-last.checkpoint.pth", "stage1_model", {"pretrained": False}),
        ("second/whole_b7_lb22.93.pth", "stage2_smp_model", {"encoder_name": "tu-timm/tf_efficientnet_b7.ns_jft_in1k", "encoder_weights": None, "decoder_name": "unet", "use_coord_conv": True, "pretrained": False}),
        ("second/whole_v2_l_lb22.60.pth", "stage2_smp_model", {"encoder_name": "tu-timm/tf_efficientnetv2_l.in21k", "encoder_weights": None, "decoder_name": "unet", "use_coord_conv": True, "pretrained": False}),
        ("second/series_b6_shared_conv2d_lb23.10.pth", "stage2_lead_model", {"encoder_name": "tu-timm/tf_efficientnet_b6.ns_jft_in1k", "encoder_weights": None, "fusion_type": "shared_conv2d"}),
        ("second/series_b6_shared_conv2d_lb23.00.pth", "stage2_lead_model", {"encoder_name": "tu-timm/tf_efficientnet_b6.ns_jft_in1k", "encoder_weights": None, "fusion_type": "shared_conv2d"}),
        ("second/series_v2_l_conv3d_lb22.92.pth", "stage2_lead_model", {"encoder_name": "tu-timm/tf_efficientnetv2_l.in21k", "encoder_weights": None, "fusion_type": "conv3d"}),
        ("second/series_v2_l_conv2d_lb22.85.pth", "stage2_lead_model", {"encoder_name": "tu-timm/tf_efficientnetv2_l.in21k", "encoder_weights": None, "fusion_type": "conv2d"}),
    ]
    for path, module, kwargs in targets:
        row = {"asset": path, "module": module, "constructor": kwargs, "status": "scheduled"}
        try:
            model = importlib.import_module(module).Net(**kwargs)
            state = torch.load(root / path, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)
            row.update(status="strict_load_passed", parameters=sum(p.numel() for p in model.parameters()))
            del model, state
            gc.collect()
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
        records.append(row)
        write_json(args.out, {"schema": "kaggle-native-checkpoint-audit-v1", "environment": environment(), "assets_lock_sha256": sha256(root / "assets.lock.json"), "native_inference_completed": False, "original_helpers": helpers, "records": records})
    if any(r["status"] != "strict_load_passed" for r in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
