"""Narrow independent review of2D cross compatibility, without an API shim.

Create the historical numeric reference under the existing NumPy2.3.5:
  swarm-venv/bin/python m4/qa/review_numpy_compatibility.py --make-reference
Run the unchanged26test methods under each installed NumPy runtime:
  swarm-venv/bin/python m4/qa/review_numpy_compatibility.py --label numpy235
  PYTHONPATH=/workspace/scratch/a57599a3e70e/m4-numpy25-runtime \
    swarm-venv/bin/python m4/qa/review_numpy_compatibility.py --label numpy254
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import unittest
import warnings
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/"receipts/numpy-compat"
OLD_SHA = "0ce6062797c5e25dd9ce3d7b26a6a9fabe7779ca72b5ea4e55fcda564bed9e87"
NEW_SHA = "421ebe98bdd9b5c18daf039837297d8592e9fdb4f937b466311033a9f6be7fe0"
OLD = HERE/"snapshots/geometry_selector-guard-v2-before-numpy-compat.py"
NEW = ROOT/"m4/geometry/geometry_selector.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_source():
    if digest(OLD) != OLD_SHA or digest(NEW) != NEW_SHA:
        raise AssertionError("Source identity changed from the narrow review")
    old = OLD.read_text()
    expected = old.replace(
        "    cross = np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])\n",
        "    edge_a = vertices[:, 1]-vertices[:, 0]\n"
        "    edge_b = vertices[:, 2]-vertices[:, 0]\n"
        "    cross = edge_a[:, 0]*edge_b[:, 1]-edge_a[:, 1]*edge_b[:, 0]\n")
    if expected == old or expected != NEW.read_text():
        raise AssertionError("Change exceeds the exact intended scalar determinant substitution")
    # Compile only the actual reviewed assignments into an independent probe.
    # This does not patch NumPy, the production function or any import.
    module = ast.parse(NEW.read_text())
    guard = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "guard_complex")
    wanted = []
    for statement in guard.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in {"edge_a", "edge_b", "cross"}:
                wanted.append(statement)
    if len(wanted) != 3:
        raise AssertionError("Unexpected cross computation")
    body = "def scalar_cross_from_reviewed_source(vertices):\n"
    for node in wanted:
        body += "    "+ast.unparse(node)+"\n"
    body += "    return cross\n"
    namespace = {"np": np}
    exec(compile(body, str(NEW)+":reviewed-assignments", "exec"), namespace)
    return namespace["scalar_cross_from_reviewed_source"]


def make_reference():
    if np.__version__ != "2.3.5":
        raise AssertionError("Historical cross reference must use the actual existing2.3.5 runtime")
    review_source()
    rng = np.random.default_rng(61023127)
    vertices = rng.normal(size=(8192, 3, 2))
    vertices *= np.power(10., rng.uniform(-140, 140, (len(vertices), 1, 1)))
    exact = []
    for sign in [-1., 1.]:
        for eps in [-1e-12, 0., 1e-12]:
            exact.append([[0., 0.], [1., 0.], [1., sign*1e-10*(1+eps)]])
        exact.extend([[[0., 0.], [sign, 0.], [0., 0.]],
                      [[0., -0.], [-0., 0.], [0., -0.]],
                      [[0., 0.], [1e-160, 0.], [0., sign*1e-160]]])
    vertices = np.concatenate((vertices, np.array(exact)), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        historical = np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])
    if not np.isfinite(historical).all():
        raise AssertionError("Reference unexpectedly nonfinite")
    np.savez_compressed(OUT/"historical-cross-reference.npz", vertices=vertices, expected=historical)
    receipt = {"schema": "m4-historical-cross-reference-v1", "numpy": np.__version__,
        "python": platform.python_version(), "historical_source_sha256": OLD_SHA,
        "reference_sha256": digest(OUT/"historical-cross-reference.npz"),
        "float64_triangles": len(vertices), "source": "Actual np.cross2D in NumPy2.3.5, no shim or monkeypatch",
        "coverage": "Finite broad-exponent random triangles, near-degenerate orientation thresholds, exact degeneracy, signed zero and subnormal products"}
    (OUT/"historical-cross-reference.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


def load_test_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(label):
    reviewed_cross = review_source()
    reference_path = OUT/"historical-cross-reference.npz"
    reference_receipt = json.loads((OUT/"historical-cross-reference.json").read_text())
    if digest(reference_path) != reference_receipt["reference_sha256"]:
        raise AssertionError("Historical reference changed")
    with np.load(reference_path, allow_pickle=False) as archive:
        actual = reviewed_cross(archive["vertices"])
        expected = archive["expected"]
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        triangles = len(actual)
    previous_receipts = {name: digest(HERE/"receipts"/name) for name in
        ["independent-component-tests.json", "independent-axis-area-tests.json"]}
    source_paths = [NEW, ROOT/"m4/sampling/streaming_sampler.py", ROOT/"m4/sampling/pngio.py",
        ROOT/"m4/sampling/axis_aligned_area.py", Path(__file__)]
    before = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    component = load_test_module("m4_numpy_compat_components", HERE/"independent_component_tests.py")
    axis = load_test_module("m4_numpy_compat_axis", HERE/"independent_axis_area_tests.py")
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(component),
                               unittest.defaultTestLoader.loadTestsFromModule(axis)])
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    stable = all(digest(ROOT/path) == value for path, value in before.items())
    unchanged_receipts = all(digest(HERE/"receipts"/name) == value for name, value in previous_receipts.items())
    receipt = {"schema": "m4-independent-numpy-compatibility-review-v1", "label": label,
        "passed": result.wasSuccessful() and stable and unchanged_receipts and result.testsRun == 26,
        "numpy": np.__version__, "numpy_path": str(Path(np.__file__).resolve()),
        "python": platform.python_version(), "test_methods": result.testsRun,
        "historical_float64_cross_triangles": triangles, "cross_bit_mismatches": 0,
        "source_diff_exactly_the_reviewed_substitution": True,
        "sources_stable_during_run": stable, "historical_approval_receipts_unchanged": unchanged_receipts,
        "source_sha256": before, "prior_receipt_sha256": previous_receipts,
        "reference_sha256": digest(reference_path),
        "elapsed_seconds": time.perf_counter()-started,
        "failures": [{"test": str(t), "trace": text} for t, text in result.failures],
        "errors": [{"test": str(t), "trace": text} for t, text in result.errors],
        "component_notes": component.NOTES, "axis_notes": axis.NOTES,
        "limits": ["Compatibility repair only, not a new methodology or fresh clinical result",
            "No NumPy API shim, dependency pin or monkeypatch to replace np.cross",
            "26existing tests include their original PNG failure injections",
            "Finite IEEE754 parity controls and literal source diff support equivalence, not all possible arrays"]}
    target = OUT/(label+".json")
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"receipt": str(target), "passed": receipt["passed"],
        "numpy": np.__version__, "tests": result.testsRun, "triangles": triangles,
        "historical_receipts_unchanged": unchanged_receipts}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-reference", action="store_true")
    parser.add_argument("--label")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    if args.make_reference:
        make_reference()
    elif args.label:
        raise SystemExit(run(args.label))
    else:
        parser.error("Provide --make-reference or --label")
