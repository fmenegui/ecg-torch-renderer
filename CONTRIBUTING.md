# Contributing

Keep contracts explicit: standard lead order, mV input, sampling rates, device,
output shapes, visibility and box coordinates. Add meaningful regression checks
when changing these contracts or numerical behavior.

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python tools/check_version.py
python -m unittest discover -s tests -v
python examples/segmentation.py --steps 3
npm install --prefix documentation
python documentation/build.py
python documentation/validate.py
```

Use clear names and small functions. Keep file decoding, font initialization,
NumPy conversion and encoding out of forward. Avoid host scalar reads during
rendering. Measure changes with identical inputs, seeds and a warmed runtime.

Bump the version in `pyproject.toml`, `ecg_gpu/__init__.py` and a new top
`CHANGELOG.md` entry together. For a release archive, generate provenance with
`python tools/source_manifest.py --out build/MANIFEST.json`.

Update documentation and verification when behavior changes. Use artificial
signals for examples. Real data and assets require their own permissions,
patient-disjoint evaluation and provenance. Describe the concrete behavior
change and relevant validation in a pull request. Record GPU tests as skipped
when hardware is unavailable. Do not substitute projected FPS or clinical
performance claims for measured evidence.
