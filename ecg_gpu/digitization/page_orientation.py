"""Optional semantic page orientation from readable ECG text, never grid truth.

Tesseract is an external, explicitly requested offline dependency. Raw OCR and
all competing rotations are retained. Low or ambiguous text evidence leaves
orientation unresolved. The RGB used for rectification is never binarized.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np


def _text_evidence(tsv):
    """Bound repeated tokens so a grid/waveform cannot win by token count."""
    strong, weak, headers = {}, {}, {}
    words = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        token = row.get("text", "").strip()
        try:
            confidence = float(row.get("conf", -1)) / 100
        except ValueError:
            continue
        if not token or confidence < 0.35:
            continue
        normalized = re.sub(r"[^a-zA-Z0-9]", "", token).upper()
        words.append({"text": token, "confidence": confidence})
        if re.fullmatch(r"V[1-6]|AV[LRF]", normalized):
            strong[normalized] = max(confidence, strong.get(normalized, 0.0))
        elif normalized in ("I", "II", "III") and confidence >= 0.6:
            weak[normalized] = max(confidence, weak.get(normalized, 0.0))
        elif normalized in (
            "ECG",
            "PATIENT",
            "DATE",
            "AGE",
            "MALE",
            "FEMALE",
            "NAME",
            "RATE",
            "HOSPITAL",
            "PACIENTE",
            "DATA",
            "IDADE",
        ):
            headers[normalized] = max(confidence, headers.get(normalized, 0.0))
    score = (
        3 * sum(strong.values()) + 0.5 * sum(weak.values()) + 2 * sum(headers.values())
    )
    return {
        "score": score,
        "strong_labels": strong,
        "weak_labels": weak,
        "header_words": headers,
        "recognized_words": words,
    }


def choose_text_rotation(evidence):
    """Require redundant recognized labels and a clear margin over rotations."""
    if len(evidence) != 4:
        raise ValueError("Evidence for all four quarter rotations required")
    order = sorted(range(4), key=lambda k: (-evidence[k]["score"], k))
    best, runner = order[:2]
    top = evidence[best]
    redundant = len(top["strong_labels"]) >= 2 or len(top["header_words"]) >= 2
    margin = top["score"] - evidence[runner]["score"]
    complete = not any(item.get("error") for item in evidence)
    accepted = complete and redundant and top["score"] >= 3.0 and margin >= 1.5
    return {
        "status": "resolved-from-text" if accepted else "unresolved",
        "quarter_turns_ccw": best if accepted else 0,
        "best_proposal_ccw": best,
        "score_margin": margin,
        "reason": (None if accepted else "insufficient or ambiguous readable ECG text")
        if complete
        else "incomplete OCR execution",
        "semantic_orientation_inferred": accepted,
        "reference_used": False,
        "rule": "at least two distinct strong lead labels or two header words, score>=3, margin>=1.5",
    }


def infer_text_orientation(rgb, *, artifact_dir=None, timeout_seconds=20.0):
    """Run pinned-by-receipt local OCR on separate maximum-channel inputs."""
    import cv2

    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("uint8 HWC RGB required")
    if max(rgb.shape[:2]) > 8192 or min(rgb.shape[:2]) < 2:
        raise ValueError("Image outside bounded OCR dimensions")
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Positive finite OCR timeout required")
    binary = shutil.which("tesseract")
    if binary is None:
        raise RuntimeError(
            "Text orientation requires an installed tesseract executable"
        )
    version = subprocess.run(
        [binary, "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_seconds,
    ).stdout.splitlines()[0]
    languages = subprocess.run(
        [binary, "--list-langs"],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_seconds,
    ).stdout
    match = re.search(r'"([^"]+)"', languages)
    english = Path(match.group(1)) / "eng.traineddata" if match else None
    english_sha256 = (
        hashlib.sha256(english.read_bytes()).hexdigest()
        if english is not None and english.exists()
        else None
    )
    # The maximum color channel suppresses common red/green grid ink. This is
    # OCR-only preprocessing, original RGB remains the sampling source.
    gray = rgb.max(axis=-1)
    scale = min(2.0, 2048 / max(gray.shape))
    enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    prepared = cv2.adaptiveThreshold(
        enlarged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 12
    )
    evidence = []
    with tempfile.TemporaryDirectory(prefix="ecg-orientation-") as temporary:
        root = Path(temporary)
        for quarter in range(4):
            filename = root / f"rotation-{quarter}.png"
            cv2.imwrite(str(filename), np.rot90(prepared, quarter))
            environment = dict(os.environ, OMP_THREAD_LIMIT="1")
            try:
                result = subprocess.run(
                    [
                        binary,
                        str(filename),
                        "stdout",
                        "-l",
                        "eng",
                        "--psm",
                        "11",
                        "tsv",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env=environment,
                )
                tsv, stderr = result.stdout, result.stderr
                error = (
                    "Tesseract exited with code " + str(result.returncode)
                    if result.returncode
                    else None
                )
            except subprocess.TimeoutExpired:
                tsv, stderr, error = "", "", "Tesseract exceeded the rotation timeout"
            item = _text_evidence(tsv)
            if error:
                item["error"] = error
            evidence.append(item)
            if artifact_dir is not None:
                folder = Path(artifact_dir)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"rotation-{quarter}.tsv").write_text(tsv)
                (folder / f"rotation-{quarter}.stderr.txt").write_text(stderr)
    return {
        **choose_text_rotation(evidence),
        "candidates": evidence,
        "tesseract_version": version,
        "tesseract_binary_sha256": hashlib.sha256(
            Path(binary).read_bytes()
        ).hexdigest(),
        "english_traineddata_sha256": english_sha256,
        "rgb_sha256": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
        "preprocessing": "maximum RGB channel, <=2048px enlargement, adaptive threshold",
        "text_rule_is_calibrated_probability": False,
    }


def rotate_lattice_ids(topology, quarter_turns_ccw):
    """Rotate relative coordinates while retaining the same observed cells."""
    if type(quarter_turns_ccw) is not int or quarter_turns_ccw not in range(4):
        raise ValueError("Integer quarter rotation in [0,3] required")
    ids = np.asarray(topology["ids"], int).reshape(-1, 2)
    rotation = np.array([[0, 1], [-1, 0]], dtype=int)
    matrix = np.linalg.matrix_power(rotation, quarter_turns_ccw)
    transformed = (ids[:, ::-1] @ matrix.T)[:, ::-1]
    offset = transformed.min(0) if len(ids) else np.zeros(2, int)
    transformed -= offset
    cells = []
    for r, c in topology["cells"]:
        corners = np.array([[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]])
        target = (corners @ matrix.T)[:, ::-1] - offset
        cells.append(target.min(0).tolist())
    return dict(
        topology,
        ids=transformed.tolist(),
        cells=cells,
        page_quarter_turns_ccw=quarter_turns_ccw,
        semantic_coordinate_rotation=matrix.tolist(),
    )
