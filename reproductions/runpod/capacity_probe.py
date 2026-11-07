"""Original full batch/patch capacity and CUDA RNG checkpoint controls."""

import argparse
import json
from pathlib import Path
import sys
import time

from reproductions.common import environment, verify_source, write_json
from reproductions.npj.protocol import COMMIT, training_config
from reproductions.npj.train import _resolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    verify_source(args.upstream, COMMIT)
    sys.path.insert(0, str(Path(args.upstream).resolve()))
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    config = training_config(args.upstream, "paper_table4", "/capacity/train", "/capacity/val")
    result = {"schema": "npj-cuda-capacity-control-v1", "environment": environment(), "input": [12, 3, 1024, 1024], "synthetic_engineering_control": True, "compiled": False, "full_training_completed": False}
    torch.manual_seed(1973)
    state = torch.cuda.get_rng_state_all()
    expected = torch.rand(8, device="cuda")
    torch.save({"cuda_rng": state}, output / "rng.pt")
    loaded = torch.load(output / "rng.pt", map_location="cuda", weights_only=True)
    torch.cuda.set_rng_state_all([item.cpu() for item in loaded["cuda_rng"]])
    result["cuda_rng_checkpoint_roundtrip_equal"] = torch.equal(expected, torch.rand(8, device="cuda"))
    started = time.perf_counter()
    try:
        model_cfg = config["MODEL"]
        model = _resolve(model_cfg["class_path"])(**model_cfg["KWARGS"]).cuda()
        opt_cfg = config["TRAIN"]["OPTIMIZER"]
        optimizer = _resolve(opt_cfg["class_path"])(model.parameters(), **opt_cfg["KWARGS"])
        loss_cfg = config["TRAIN"]["CRITERION"]
        criterion = _resolve(loss_cfg["class_path"])(**loss_cfg["KWARGS"])
        image = torch.rand((12, 3, 1024, 1024), device="cuda")
        labels = torch.randint(4, (12, 1024, 1024), device="cuda")
        target = torch.nn.functional.one_hot(labels, 4).permute(0, 3, 1, 2).float()
        torch.cuda.reset_peak_memory_stats()
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast("cuda"):
            loss = criterion(model(image), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        result.update(full_shape_feasible=True, loss=float(loss.detach()), peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except RuntimeError as error:
        result.update(full_shape_feasible=False, error=str(error), error_type=type(error).__name__)
    result["seconds"] = time.perf_counter() - started
    write_json(output / "receipt.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
