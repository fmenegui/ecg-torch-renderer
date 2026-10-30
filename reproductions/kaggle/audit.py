"""Audit original notebook versus the author's later training-code refactor."""

import argparse
import ast
import json
from pathlib import Path
from typing import List

import numpy as np
from scipy import signal

from reproductions.common import sha256, verify_source, write_json
from reproductions.kaggle.replay import adapt, notebook_code
from reproductions.kaggle.train import COMMIT


def extract_function(code, name, namespace):
    matches = [node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError("Native function identity changed")
    tree = ast.Module(body=matches, type_ignores=[])
    exec(compile(tree, "original-author-function", "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = verify_source(args.upstream, COMMIT)
    document = json.loads(Path(args.notebook).read_text())
    code, source_digest = notebook_code(document)
    notebook_decoder = extract_function(code, "pixel_to_series_exp", {"np": np, "signal": signal})
    public_decoder = extract_function((Path(args.upstream) / "predict.py").read_text(), "pixel_to_series", {"np": np, "scipy_signal": signal, "List": List})
    pixels = np.zeros((4, 32, 16), np.float32)
    pixels[:, 8] = 0.75
    pixels[:, 10] = 0.25
    expected = notebook_decoder(pixels.copy(), [16] * 4, 16)
    refactor = public_decoder(pixels.copy(), [16] * 4, 16)
    _, changes = adapt(document, {"/kaggle/working/outputs": "/tmp/retained-probe"})
    write_json(args.out, {"schema": "kaggle-author-source-audit-v1", "source": source, "notebook_version": 6, "notebook_source_sha256": source_digest, "public_predict_sha256": sha256(Path(args.upstream) / "predict.py"), "native_decoder_bimodal_probe": {"notebook_mean_y": float(expected.mean()), "refactor_mean_y": float(refactor.mean()), "equal": bool(np.array_equal(expected, refactor))}, "native_notebook_adaptation_contract": changes, "interpretation": "The later public predict.py and original final notebook do not implement identical waveform decoding on this finite control. This does not establish a leaderboard-score change.", "gpu_inference_completed": False, "training_completed": False, "leaderboard_reproduced": False})


if __name__ == "__main__":
    main()
