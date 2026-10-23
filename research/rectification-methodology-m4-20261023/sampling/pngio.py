"""Verified atomic PNG publication, with default refusal to replace a file."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np
from PIL import Image, ImageFile


@dataclass(frozen=True)
class PngReceipt:
    path: str
    mode: str
    shape: tuple[int, ...]
    encoded_sha256: str
    decoded_sha256: str
    encoded_bytes: int
    replaced_existing: bool
    directory_fsynced: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_png_chunks(path: Path):
    """Require all chunk payloads/CRCs and a terminal IEND, even after IDAT."""
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("not a PNG")
        first, has_data = True, False
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("truncated PNG before IEND")
            length, kind = struct.unpack(">I4s", header)
            if first and (kind != b"IHDR" or length != 13):
                raise ValueError("invalid first PNG chunk")
            first = False
            remaining, crc = length, zlib.crc32(kind)
            while remaining:
                block = stream.read(min(remaining, 1024*1024))
                if not block:
                    raise ValueError("truncated PNG chunk payload")
                remaining -= len(block)
                crc = zlib.crc32(block, crc)
            tail = stream.read(4)
            if len(tail) != 4 or struct.unpack(">I", tail)[0] != crc & 0xffffffff:
                raise ValueError("missing or invalid PNG CRC")
            has_data |= kind == b"IDAT"
            if kind == b"IEND":
                if length or not has_data or stream.read(1):
                    raise ValueError("invalid PNG termination")
                return


def verify_png(path: str | Path, expected_u8: np.ndarray) -> tuple[str, str]:
    """Check chunks, fully decode, and compare exact mode/shape/pixel values."""
    if (not isinstance(expected_u8, np.ndarray) or expected_u8.dtype != np.uint8 or
            expected_u8.ndim != 3 or expected_u8.shape[2] not in (3, 4) or
            min(expected_u8.shape[:2]) < 1):
        raise ValueError("expected pixels must be nonempty uint8 RGB/RGBA")
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ValueError("strict PNG verification requires LOAD_TRUNCATED_IMAGES=False")
    path = Path(path)
    _check_png_chunks(path)
    mode = "RGB" if expected_u8.shape[2] == 3 else "RGBA"
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        if image.mode != mode:
            raise ValueError("decoded PNG mode differs from input")
        decoded = np.array(image)
    if decoded.shape != expected_u8.shape or not np.array_equal(decoded, expected_u8):
        raise ValueError("decoded PNG pixels differ from input")
    return _sha256(path), hashlib.sha256(decoded.tobytes()).hexdigest()


def _fsync_directory(path: Path) -> bool:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        return False
    return True


def write_png_atomic(
    path: str | Path,
    image_u8: np.ndarray,
    *,
    expected_existing_sha256: str | None = None,
) -> PngReceipt:
    """Validate a complete PNG before atomic publication in its directory.

    Default publication uses a hard link, which atomically refuses an existing
    target even if another process creates it during encoding. Explicit
    replacement requires the expected current SHA256. Writers using this API
    coordinate through an exclusive adjacent lock and recheck before replace.
    External writers ignoring the lock can still race a guarded replacement.
    Stale locks fail closed and are never silently removed by a different call.

    A returned receipt means publication succeeded. Directory fsync support is
    reported separately. No promise is made against later external corruption.
    """
    if (not isinstance(image_u8, np.ndarray) or image_u8.dtype != np.uint8 or
            image_u8.ndim != 3 or image_u8.shape[2] not in (3, 4) or
            min(image_u8.shape[:2]) < 1):
        raise ValueError("image must be nonempty uint8 RGB/RGBA")
    path = Path(path).absolute()
    if path.suffix.lower() != ".png":
        raise ValueError("destination must end in .png")
    if expected_existing_sha256 is not None and (
            len(expected_existing_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in expected_existing_sha256)):
        raise ValueError("expected SHA256 must be 64 lowercase hex digits")
    lock = path.with_name("."+path.name+".write-lock")
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    temporary = None
    try:
        if path.is_symlink():
            raise ValueError("refuse to replace a symbolic link")
        if expected_existing_sha256 is None:
            if path.exists():
                raise FileExistsError(path)
        elif not path.is_file() or _sha256(path) != expected_existing_sha256:
            raise ValueError("existing PNG SHA256 does not match replacement guard")
        with tempfile.NamedTemporaryFile(mode="w+b", prefix="."+path.name+".",
                                         suffix=".tmp.png", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            Image.fromarray(image_u8).save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        encoded_sha, decoded_sha = verify_png(temporary, image_u8)
        encoded_bytes = temporary.stat().st_size
        if expected_existing_sha256 is None:
            os.link(temporary, path)
            temporary.unlink()
        else:
            if path.is_symlink() or not path.is_file() or _sha256(path) != expected_existing_sha256:
                raise ValueError("existing PNG changed before replacement")
            os.replace(temporary, path)
        temporary = None
        synced = _fsync_directory(path.parent)
        return PngReceipt(str(path), "RGB" if image_u8.shape[2] == 3 else "RGBA",
                          tuple(image_u8.shape), encoded_sha, decoded_sha, encoded_bytes,
                          expected_existing_sha256 is not None, synced)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        os.close(lock_fd)
        lock.unlink()
