"""Audited training harness using original npj networks, losses and transforms.

This is a harness replication, not a claim of bit-identical author training.
Full profiles preserve documented hyperparameters. --smoke is an explicitly
reduced engineering test. Resume is supported at completed epoch boundaries.
"""

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from reproductions.common import environment, new_run, require_cuda, sha256, verify_source, write_json
from reproductions.npj.data_identity import validate_receipt_source
from reproductions.npj.protocol import COMMIT, training_config


def optimizer_state(optimizer):
    if hasattr(optimizer, "muon"):
        return {"kind": "AdamMuon", "muon": optimizer.muon.state_dict() if optimizer.muon else None, "adam": optimizer.adam.state_dict() if optimizer.adam else None}
    return {"kind": "standard", "state": optimizer.state_dict()}


def restore_optimizer(optimizer, state):
    if state["kind"] == "standard":
        optimizer.load_state_dict(state["state"])
        return
    groups = []
    for key in ["muon", "adam"]:
        inner = getattr(optimizer, key)
        if (inner is None) != (state[key] is None):
            raise ValueError("Changed optimizer groups")
        if inner:
            inner.load_state_dict(state[key])
            groups.extend(inner.param_groups)
    optimizer.param_groups = groups


def _resolve(name):
    module, attribute = name.rsplit(".", 1)
    return getattr(importlib.import_module(module), attribute)


class PauseTraining(Exception):
    """Discard an unfinished epoch and resume its saved starting state later."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--profile", choices=["paper_table4", "public_config", "lead_text_public"], required=True)
    parser.add_argument("--train-dir")
    parser.add_argument("--validation-dir")
    parser.add_argument("--data-receipt", help="Required original full-data extraction receipt outside engineering smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1973)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--invocation-deadline", type=float, help="Unix time to pause this invocation without changing the full training profile")
    args = parser.parse_args()
    require_cuda(args.device)
    upstream = Path(args.upstream).resolve()
    verify_source(upstream, COMMIT)
    sys.path.insert(0, str(upstream))
    import torch
    from torch.utils.data import DataLoader, Subset
    from yacs.config import CfgNode
    from src.utils import get_data_loaders
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out).resolve() if args.resume else new_run(args.out)
    if args.resume and not (out / "last.pt").is_file():
        raise ValueError("No completed-epoch checkpoint")
    config = training_config(upstream, args.profile, args.train_dir, args.validation_dir, args.device)
    torch.backends.cudnn.benchmark = bool(config["TRAIN"].get("CUDNN_BENCHMARK", False))
    data_identity = None
    if args.profile != "lead_text_public" and not args.smoke:
        if not args.data_receipt:
            raise ValueError("Full main-model reproduction requires the original complete data receipt")
        data_identity = json.loads(Path(args.data_receipt).read_text())
        if data_identity.get("schema") != "npj-public-data-extraction-v1" or not data_identity.get("complete_public_release"):
            raise ValueError("Incomplete or unknown development-data release")
        validate_receipt_source(data_identity)
        dataset_root = Path(args.data_receipt).resolve().parent
        if Path(args.train_dir).resolve() != dataset_root / "train" or Path(args.validation_dir).resolve() != dataset_root / "val":
            raise ValueError("Training paths differ from the extracted original split")
        for record in data_identity["files"]:
            if sha256(dataset_root / record["file"]) != record["sha256"]:
                raise ValueError("Changed image/mask in the original development data")
    if args.smoke:
        config["TRAIN"]["KWARGS"]["epochs"] = 1
        config["TRAIN"]["COMPILE"] = False
        config["DATALOADER"]["KWARGS"].update(batch_size=1, num_workers=0, pin_memory=False, drop_last=False)
        config["DATALOADER"]["KWARGS"].pop("prefetch_factor", None)
        config["DATALOADER"]["KWARGS"].pop("pin_memory_device", None)
        if args.profile == "lead_text_public":
            for split in ["TRAIN", "VAL", "TEST"]:
                config["DATASET"][split]["KWARGS"]["num_samples"] = 2
        else:
            for split in ["TRAIN", "VAL"]:
                transforms = config["DATASET"][split]["TRANSFORM"]["KWARGS"]["transform_config"]
                for transform in transforms:
                    if transform["class_path"].endswith("RandomCrop"):
                        transform["KWARGS"]["size"] = [256, 256]
    previous = out / "effective-config.json"
    if args.resume and json.loads(previous.read_text()) != config:
        raise ValueError("Resume configuration changed")
    write_json(previous, config)
    started = time.perf_counter()
    receipt = {"schema": "npj-training-v1", "status": "started", "profile": args.profile, "seed": args.seed, "engineering_smoke": args.smoke, "source_commit": COMMIT, "environment": environment(), "harness_changes": ["explicit torch.amp.GradScaler device and arguments", "no storage of epoch predictions", "atomic completed-epoch checkpoint with both inner optimizer states and RNG", "explicit seed and effective configuration"], "paper_cohort_reproduction_completed": False}
    receipt["data_receipt_sha256"] = sha256(args.data_receipt) if args.data_receipt else None
    write_json(out / "receipt.json", receipt)
    try:
        model_config = config["MODEL"]
        model = _resolve(model_config["class_path"])(**model_config["KWARGS"]).to(args.device)
        receipt["parameters"] = sum(p.numel() for p in model.parameters())
        opt_config = config["TRAIN"]["OPTIMIZER"]
        optimizer = _resolve(opt_config["class_path"])(model.parameters(), **opt_config["KWARGS"])
        lr_config = config["TRAIN"]["LR_SCHEDULER"]
        scheduler = _resolve(lr_config["class_path"])(optimizer, **lr_config["KWARGS"])
        loss_config = copy.deepcopy(config["TRAIN"]["CRITERION"])
        if "weight" in loss_config["KWARGS"]:
            loss_config["KWARGS"]["weight"] = torch.tensor(loss_config["KWARGS"]["weight"], device=args.device)
        criterion = _resolve(loss_config["class_path"])(**loss_config["KWARGS"])
        train_loader, val_loader, _ = get_data_loaders(CfgNode(config["DATASET"]), config["DATALOADER"]["KWARGS"])
        if args.smoke:
            train_loader = DataLoader(Subset(train_loader.dataset, range(min(2, len(train_loader.dataset)))), batch_size=1, shuffle=False)
            val_loader = DataLoader(Subset(val_loader.dataset, range(min(1, len(val_loader.dataset)))), batch_size=1, shuffle=False)
        if not len(train_loader) or not len(val_loader):
            raise ValueError("Empty training or validation loader")
        scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
        start_epoch, best = 0, float("inf")
        if args.resume:
            state = torch.load(out / "last.pt", weights_only=False, map_location=args.device)
            model.load_state_dict(state["model"])
            restore_optimizer(optimizer, state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if args.device.startswith("cuda"):
                torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])
            start_epoch, best = state["completed_epochs"], state["best_validation_loss"]
        receipt["train_images"] = len(train_loader.dataset)
        receipt["validation_images"] = len(val_loader.dataset)
        if config["TRAIN"].get("COMPILE"):
            forward_model = torch.compile(model)
        else:
            forward_model = model
        for epoch in range(start_epoch, config["TRAIN"]["KWARGS"]["epochs"]):
            metrics = {"epoch": epoch + 1}
            for split, loader in [("train", train_loader), ("validation", val_loader)]:
                model.train(split == "train")
                total, count = 0.0, 0
                for image, target in loader:
                    if args.invocation_deadline and time.time() >= args.invocation_deadline:
                        raise PauseTraining()
                    image, target = image.to(args.device), target.to(args.device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(split == "train"):
                        with torch.autocast(device_type="cuda", enabled=args.device.startswith("cuda")):
                            loss = criterion(forward_model(image), target)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training loss")
                        if split == "train":
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                    total += float(loss.detach()) * len(image)
                    count += len(image)
                metrics[split + "_loss"] = total / count
            improved = metrics["validation_loss"] < best
            best = min(best, metrics["validation_loss"])
            state = {"model": model.state_dict(), "optimizer": optimizer_state(optimizer), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "completed_epochs": epoch + 1, "best_validation_loss": best, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None}
            torch.save(state, out / "last.pt.tmp")
            os.replace(out / "last.pt.tmp", out / "last.pt")
            if improved:
                torch.save(model.state_dict(), out / "best-model.pt.tmp")
                os.replace(out / "best-model.pt.tmp", out / "best-model.pt")
            with (out / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            receipt.update(completed_epochs=epoch + 1, last_epoch=metrics)
            write_json(out / "receipt.json", receipt)
        receipt["status"] = "completed_engineering_smoke" if args.smoke else "completed_training"
    except PauseTraining:
        receipt.update(status="paused_training", resume_requires_completed_epoch=True)
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["wall_seconds_this_invocation"] = time.perf_counter() - started
        write_json(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
