# Handoff: state of the work and how to continue

Read this first in a new session, then `STATE_OF_THE_ART_ROADMAP.md` for item
status and acceptance criteria. Last updated with 0.32.0.

## Where things stand

The goal is the ECG image generator whose synthetic-only training transfers
best to real ECG images, shown on public benchmarks. Since 0.13.0 the renderer
gained fifteen opt-in realism and supervision options. **None has been measured
on real images yet.** The next step is measurement, not more features.

| Option (`RenderConfig`) | What it does | Version |
| --- | --- | --- |
| `page_model="physical"` | Fixed paper width; speed sets the time window; mm stroke widths; feed error; trace skew | 0.14, 0.25 |
| `stroke="isotropic"` | Constant stroke width at any slope | 0.15 |
| `lead_marks=True` | Lead-switch marks at column boundaries | 0.16 |
| `scan_modes=True` | Color, grayscale, bilevel scans at 72-600 dpi; lid shadow | 0.17, 0.30 |
| `device_filters=True` | Prints recorder-filtered and clipped signal; reference stays the source | 0.18, 0.28 |
| `report_text=True` | Header, measurement and footer text; rate from the printed lead II | 0.19 |
| `phone_isp=True` | Chromatic aberration, sharpening halos, glare | 0.20 |
| `coordinate_targets=True` | Dense per-pixel time and mV maps | 0.21 |
| `print_technology=True` | Laser halftone, photocopy, fax | 0.23 |
| `archive_damage=True` | Mold and ironing scorch | 0.26 |
| `page_corners=True` | Observed sheet corners | 0.27 |
| `calibration_target=True` | Calibration pulse mask | 0.28 |
| `office_marks=True` | Barcodes, QR-style labels, ink stamps | 0.29 |
| `perspective_relief=True` | Pinhole parallax for lifted paper | 0.31 |
| `text_target=True` | All printed text as one target | 0.32 |

Measurement tools: `ecg-waveform-benchmark` (0.22), the grid benchmark
(0.12) and the sim-to-real notebook below (0.32). The digitization research
line from #11-#29 (`ecg_gpu.digitization`, `reproductions/`) was integrated
in 0.24.

## Next step: run the sim-to-real notebook

`notebooks/sim2real_benchmark.ipynb` (Colab, GPU runtime) trains the
Open-ECG-Digitizer segmentation network with the same budget on pairs from
each recipe (`legacy`, `full`, optional `full-no-<flag>` ablations), digitizes
the same ECG-Image-Database D0 images with the native npj pipeline and scores
them against PTB-XL printed windows. Results go to Google Drive under
`MyDrive/ecg-torch-renderer/sim2real/<run>/`: `summary.md`, per-arm
`scores.json`, receipts, sample pairs and checkpoints.

When results exist, read `summary.md` and the per-capture rows. Decide from
them which options become defaults, then run ablations for the options that
matter. One seed and 40 patients is a pilot: differences inside the
patient-bootstrap intervals are unresolved.

Known assumptions to revisit with results: calibration pulses are labelled as
signal in npj masks (`--pulse-class`); training uses the upstream
augmentations without `torch.compile`; the `george-moody-2024` inference
preset is used for D0 images.

## Conventions every change follows

- New behaviour is opt-in through a `RenderConfig` flag. When the flag is off,
  the legacy random stream, specification and outputs are unchanged. New
  controls are drawn after all existing ones, and `specification()` pops the
  flag when it is off so saved layer states still load.
- Per-record controls are validated with `torch._assert_async` (no host
  reads). No `.item()`, NumPy or CPU work in the training path.
- Observation-only effects never change geometric targets. Waveform targets
  change only where documented (printed filters, clipping).
- The renderer never synthesizes diagnoses: no interpretation statements.
- The frozen v0.2.1 raster fixture is never regenerated.
- Each feature has analytic tests against an independent reference.
- Version bumps touch `pyproject.toml`, `ecg_gpu/__init__.py` and a new top
  `CHANGELOG.md` entry together (`tools/check_version.py` enforces it). Update
  the roadmap status line in the same change.

## Workflow notes

- The test suite needs PyTorch. If the session cannot install it, CI is the
  test run: push a branch, open a pull request, wait for every workflow run of
  the `pull_request` event to succeed, then merge. NumPy-only modules (for
  example `examples/sim2real_npj.py`, `ecg_gpu/waveform_evaluation.py`) can be
  tested locally by registering a stub `ecg_gpu` package path before import.
- `gh` works through the session proxy for REST calls; GraphQL commands such
  as `gh pr create` are blocked. Use `gh api repos/<owner>/<repo>/pulls -X POST`.
- Stacked pull requests: retarget each to `main` before merging it.
- `optional-integrations` once failed on a bit-exact Mitsuba comparison run
  with two threads; it now runs single-threaded (#48). A new failure there is
  a real failure.

## Waiting on the owner

- GPU runs: the notebook above, and the performance profile in
  `notebooks/colab_gpu_profile.ipynb` (roadmap item 29).
- PyPI credentials (item 30), clinician review (item 4), a paired real dataset
  from physical recorders (item 5) and a licensed background bank (item 24).
