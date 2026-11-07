# Paired nonlinear dewarping diagnostic, 2026-10-21

The published competition configuration disables nonlinear grid dewarping.
This experiment enables that existing upstream component without changing
neural weights, to test whether it rescues folded-paper transfer.

Eight fresh calls compare four identical RGB images, source signal, configured
paper gains, layout catalog, CPU device, Torch stack and float32 precision.
The upstream commit and both checkpoint hashes are verified by the existing
loader. Ground-truth waveforms are used only for evaluation. Geometry fitting
uses the model's grid probability map, not generator grid coordinates.
No neural network is trained. No GPU resource is created.

| Case | Disabled SNR dB / coverage | Enabled SNR dB / coverage | Raw RMSE mV disabled / enabled |
| --- | ---: | ---: | ---: |
| six-rows | 21.06 / 92.57% | 13.86 / 99.68% | 0.0377 / 0.0578 |
| portrait | 6.29 / 98.31% | -2.50 / 91.30% | 0.0875 / 0.1792 |
| offline-flat | 18.26 / 99.95% | 13.29 / 79.94% | 0.0753 / 0.0727 |
| offline-fold | -7.72 / 98.20% | -6.68 / 97.75% | 0.6249 / 0.4371 |

The folded case improves by only 1.05 dB and remains severely distorted.
Enabling the correction worsens conditional SNR in the other three cases.
For flat paper, it also lowers printed coverage from 99.95% to 79.94% and
introduces 19.90% output outside printed support, despite selecting the correct
layout family. A slightly smaller raw overlap RMSE there compares different
observed supports and cannot establish better reconstruction.

These four diagnostic cases do not support enabling dewarping globally. They
also do not prove that nonlinear dewarping is ineffective on real images or
that this component alone explains every failure. Preserve the published
configuration. Next isolate geometric errors with known-grid rectification as
an explicitly labeled oracle control, then replace oracle coordinates with
image-only predicted subpixel grid points. Keep trace and text coordinates
consistent through any replacement warp. Validate on independent real images.

SNR is the mean per-lead score on observed printed samples after up to 100 ms
alignment and DC removal, not the Kaggle metric. Coverage, raw error and
unsupported output are separate. There is one artificial source signal and
one image per configuration, so this is a mechanism diagnostic rather than a
population accuracy result. No automatic setting selector was fitted.

## Reproduction

Use the pinned upstream checkout and paper inputs documented in
`research/paper-adaptation/README.md`, then run:

```bash
python examples/digitization_dewarping_ablation.py \
  --inputs build/paper-adaptation-inputs \
  --source build/upstream/open-ecg-digitizer \
  --out build/dewarping-ablation --device cpu
```

The executor uses `torch.no_grad()` instead of `torch.inference_mode()`.
Upstream geometry fitting explicitly enables gradients to optimize grid point
positions. Inference mode prevents that and initially caused a diagnostic call
to fail. That interrupted attempt is excluded from the completed paired run.
Existing callers retain the configured upstream dewarping default when no
explicit override is supplied.

## Validation

Three new independent tests reject mismatched evidence and retain failures and
coverage separately from quality deltas. Strict E/F lint and diff checks pass.
Full repeated discovery: 348 tests, 340 pass, six skip and two errors from the
missing original `tests/fixtures/raster-v021.npz`. The fixture is not regenerated.
An initial exact-image equality failure passed both an isolated module rerun
and the complete rerun. Renderer source is unchanged. CUDA was not tested here.
Raw reports, pairing summary and validation metadata accompany this note.
