"""Source identity and run receipts shared by the reproduction experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parent


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_json(value: Any) -> Any:
    """Keep JSON interoperable, retaining nonfinite outcomes in named fields."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if hasattr(value, "item"):
        return safe_json(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(safe_json(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def verify_source(path: str | Path, expected: str, clean: bool = True) -> dict:
    path = Path(path).resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    if commit != expected:
        raise ValueError(f"Unpinned upstream: expected {expected}, got {commit}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True
    ).strip()
    if clean and status:
        raise ValueError("Upstream tracked files were modified")
    return {"commit": commit, "tracked_files_clean": not bool(status)}


def environment() -> dict:
    packages = {}
    for name in ["numpy", "scipy", "torch", "torchvision", "pillow", "opencv-python-headless", "timm", "albumentations"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda)
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["cuda_available"] = False
    return result


def new_run(path: str | Path) -> Path:
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_cuda(device: str) -> None:
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. No CPU substitution was made.")


def run_bounded(command, timeout, stdout, cwd=None):
    """Bound a POSIX process group, including data-loader and inference workers."""
    if timeout <= 0:
        raise ValueError("Positive timeout required")
    process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        raise
