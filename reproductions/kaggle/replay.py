"""Replay the original SECOND-place submission, retaining all intermediate files.

The original code remains outside our repository. This adapter changes paths,
uses serial execution on GPU 0 and removes final output deletion. It does not
replace the winning ensemble with a single model. Asset identity is mandatory.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from reproductions.common import environment, new_run, require_cuda, run_bounded, sha256, write_json

KERNEL_VERSION = 6
KERNEL = "takashisomeya/physionet-2nd-place-submission"

# Reviewed v6 demonstration-only blocks. They draw diagnostics after native
# failure handling and can reference incomplete or unbound intermediate data.
# Keep all numerical inference and accumulated failure IDs unchanged.
REVIEWED_DEMO_BLOCKS = {
    "5f6572dedb8451c8913622fe9a806a53136214d01326fc06204f0d260b7122b3",
    "6e50734c640eae8fbe7a4208a09ff51626d1669bcbb9f1113031c659ed0362a3",
    "52616c345ce7ce2aad647579a080314076d8aec865f28455d663f08e88fc435f",
}


def notebook_code(document):
    version = document.get("metadata", {}).get("currentVersionNumber")
    if version != KERNEL_VERSION:
        raise ValueError(f"Expected notebook version {KERNEL_VERSION}, got {version}")
    blob = document["blob"]
    source = blob["source"] if isinstance(blob, dict) else blob
    notebook = json.loads(source)
    cells = ["".join(c["source"]) for c in notebook["cells"] if c.get("cell_type") == "code"]
    code = "\n\n".join(cells)
    # Shell commands in these cells only install packages or list directories.
    lines = []
    for line in code.splitlines():
        if line.lstrip().startswith("!"):
            if not line.lstrip().startswith(("!ls", "!pip install")):
                raise ValueError("Unreviewed notebook shell command")
            lines.append("# Reproduction: environment setup performed separately")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n", hashlib.sha256(source.encode()).hexdigest()


class ReplayTransform(ast.NodeTransformer):
    def __init__(self, paths):
        self.paths = paths
        self.serial_replacements = []
        self.retained_outputs = 0
        self.removed_demo_blocks = []
        self.native_stage = None
        self.logged_native_failure_stages = []

    def visit_FunctionDef(self, node):
        previous = self.native_stage
        self.native_stage = node.name if node.name in ('run_stage0', 'run_stage1', 'run_stage2') else None
        try:
            return self.generic_visit(node)
        finally:
            self.native_stage = previous

    def visit_ExceptHandler(self, node):
        if self.native_stage and node.type is None:
            if len(node.body) != 1 or ast.unparse(node.body[0]) != 'local_fail_id.append(sample_id)':
                raise ValueError('Unreviewed native failure handler')
            self.logged_native_failure_stages.append(self.native_stage)
            diagnostic = ast.Expr(value=ast.Call(func=ast.Name(id='_ecg_log_native_failure', ctx=ast.Load()),
                args=[ast.Constant(value=self.native_stage), ast.Name(id='sample_id', ctx=ast.Load())], keywords=[]))
            node.body.insert(0, ast.copy_location(diagnostic, node.body[0]))
        return self.generic_visit(node)

    def visit_If(self, node):
        if ast.unparse(node.test) == "n < 10 and gpu_id == 0":
            digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            if digest not in REVIEWED_DEMO_BLOCKS:
                raise ValueError("Unreviewed numerical or diagnostic work in notebook demo branch")
            self.removed_demo_blocks.append(digest)
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value
            for old, new in sorted(self.paths.items(), key=lambda kv: -len(kv[0])):
                value = value.replace(old, str(new))
            return ast.copy_location(ast.Constant(value=value), node)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ["run_stage0_parallel", "run_stage1_parallel", "run_stage2_parallel"]:
            name = node.func.id.replace("_parallel", "")
            self.serial_replacements.append(name)
            return ast.copy_location(ast.Call(func=ast.Name(id="_ecg_serial_" + name, ctx=ast.Load()), args=[], keywords=node.keywords), node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "shutil" and node.func.attr == "rmtree":
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Name) or node.args[0].id != "OUT_DIR":
                    raise ValueError("Unreviewed deletion in submission notebook")
                self.retained_outputs += 1
                return ast.copy_location(ast.Constant(value=None), node)
        return node


def adapt(document, paths):
    code, source_hash = notebook_code(document)
    tree = ast.parse(code)
    transform = ReplayTransform(paths)
    tree = transform.visit(tree)
    ast.fix_missing_locations(tree)
    if sorted(transform.serial_replacements) != ["run_stage0", "run_stage1", "run_stage2"] or transform.retained_outputs != 1:
        raise ValueError("Notebook execution contract changed")
    # plt.show otherwise retains many large diagnostic figures on a batch worker.
    preamble = """import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: plt.close('all')
import torch
_original_load_state_dict = torch.nn.Module.load_state_dict
def _verified_load_state_dict(self, state_dict, strict=True, assign=False):
    result = _original_load_state_dict(self, state_dict, strict=False, assign=assign)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError('Native checkpoint architecture mismatch: ' + str(result))
    return result
torch.nn.Module.load_state_dict = _verified_load_state_dict
import json as _ecg_json
import traceback as _ecg_traceback
def _ecg_log_native_failure(stage, sample_id):
    try:
        print('\\nECG_NATIVE_FAILURE ' + _ecg_json.dumps({'stage': stage, 'sample_id': str(sample_id),
              'traceback': _ecg_traceback.format_exc()}), flush=True)
    except Exception:
        pass
def _ecg_serial_run_stage0(**kwargs):
    return run_stage0(**kwargs)
def _ecg_serial_run_stage1(prev_fail_ids=None, **kwargs):
    return run_stage1(prev_fail_ids=prev_fail_ids, **kwargs) + list(prev_fail_ids or [])
def _ecg_serial_run_stage2(prev_fail_ids=None, **kwargs):
    return run_stage2(prev_fail_ids=prev_fail_ids, **kwargs) + list(prev_fail_ids or [])
"""
    return preamble + ast.unparse(tree) + "\n", {"notebook_source_sha256": source_hash,
        "removed_demo_block_sha256": transform.removed_demo_blocks,
        "logged_native_failure_stages": transform.logged_native_failure_stages,
        "adaptations": ["mount path substitution", "serial GPU-0 execution of native stages preserving accumulated failure IDs", "retain native intermediate outputs", "headless diagnostic plotting", "omit only three hash-reviewed demonstration blocks, preserving native numerical code and failure handling", "log caught native exceptions without changing native failure IDs", "refuse missing or unexpected native checkpoint keys"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", required=True, help="Original API JSON for notebook v6")
    parser.add_argument("--assets", required=True, help="Contains heng/ and second/ directories")
    parser.add_argument("--assets-lock", required=True)
    parser.add_argument("--competition-data", required=True, help="Original test.csv and test/{id}.png interface")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    require_cuda("cuda")
    assets = Path(args.assets).resolve()
    lock = json.loads(Path(args.assets_lock).read_text())
    if lock.get("method") != "kaggle-second-final-six" or not lock.get("files"):
        raise ValueError("A complete pinned asset lock is required")
    expected_files = json.loads((Path(__file__).parent / "asset-sources.json").read_text())["files"]
    if set(lock["files"]) != {r["path"] for r in expected_files}:
        raise ValueError("The final six-model path requires every declared original asset")
    for relative, digest in lock["files"].items():
        if sha256(assets / relative) != digest:
            raise ValueError(f"Changed native asset: {relative}")
    import pandas as pd
    data = Path(args.competition_data).resolve()
    metadata = pd.read_csv(data / "test.csv")
    required = {"id", "lead", "number_of_rows"}
    if not required.issubset(metadata):
        raise ValueError("Missing public competition metadata")
    ids = metadata["id"].astype(str).unique().tolist()
    if not ids:
        raise ValueError("No scheduled images")
    images = {case: sha256(data / "test" / f"{case}.png") for case in ids}
    out = new_run(args.out)
    document = json.loads(Path(args.notebook).read_text())
    paths = {"/kaggle/input/hengck23-demo-submit-physionet": assets / "heng", "/kaggle/input/physionet-final-submission-models": assets / "second", "/kaggle/input/physionet-ecg-image-digitization": data, "/kaggle/working/outputs": out / "native"}
    code, changes = adapt(document, paths)
    if changes["notebook_source_sha256"] != lock.get("notebook_source_sha256"):
        raise ValueError("Notebook source identity differs from asset receipt")
    if sorted(changes["logged_native_failure_stages"]) != ['run_stage0', 'run_stage1', 'run_stage2']:
        raise ValueError('Native stage diagnostics contract changed')
    script = out / "adapted-native-submission.py"
    script.write_text(code)
    receipt = {"schema": "kaggle-native-replay-v1", "method": "second-place-final-six", "environment": environment(), "assets_lock_sha256": sha256(args.assets_lock), "images_sha256": images, "scheduled_images": len(ids), "native_submission_zeros_are_not_observed_samples": True, **changes, "adapted_script_sha256": sha256(script), "status": "started"}
    write_json(out / "receipt.json", receipt)
    try:
        with (out / "native.log").open("w") as log:
            code = run_bounded([sys.executable, str(script)], args.timeout_seconds, log, cwd=out)
        receipt.update(status="completed" if code == 0 else "failed", returncode=code)
    except subprocess.TimeoutExpired:
        receipt["status"] = "timeout"
    finally:
        receipt["cases"] = [{"case_id": case, "native_signal_file_present": (out / "native/digitalised" / f"{case}.series.npy").is_file()} for case in ids]
        if (out / "submission.csv").is_file():
            receipt["submission_sha256"] = sha256(out / "submission.csv")
        write_json(out / "receipt.json", receipt)
    if receipt["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
