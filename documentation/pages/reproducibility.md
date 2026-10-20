# Reproducibility

For independent workers/ranks, pass distinct nonnegative `stream_id` values to
the layer. Stream identity and call counters are checkpointed. The default
stream zero is reproducible and intentionally does not infer distributed state.

For offline views stable across worker assignment and dataset ordering, call
`render_seed(base_seed, source_id, view_id=...)` before creating the generator
or layer. This uses a stable digest rather than Python's salted `hash()`.

The rendering seed controls a capture. It does not assign patients to train, validation or test sets.

## Training and evaluation

| Mode | Default behavior |
| --- | --- |
| `.train()`, `randomize=True` | New capture seed for each call |
| `.eval()` | Repeat the same base capture seed |
| `randomize=False` | Repeat the capture seed in either mode |
| Explicit `generator=` | Caller owns random state and sequence |

```python
synth.eval()
a = synth(signals)
b = synth(signals)
assert torch.equal(a["image"], b["image"])
```

Fixed validation captures help compare model checkpoints. For varying batch order or size, use a recording-based seed policy in your evaluation sampler. A repeated batch seed by itself does not create a stable seed per patient across arbitrary batching.

## Save a checkpoint

```python
torch.save({
    "network": unet.state_dict(),
    "optimizer": optimizer.state_dict(),
    "synth": synth.state_dict(),
    "step": step,
}, "checkpoint.pt")
```

```python
saved = torch.load("checkpoint.pt", map_location=device, weights_only=True)
unet.load_state_dict(saved["network"])
optimizer.load_state_dict(saved["optimizer"])
synth.load_state_dict(saved["synth"])
```

The layer state includes its call counter and a configuration/assets signature. Fixed buffers and distribution buffers are also stored. A changed sample rate, renderer specification, compile setting or supervision configuration is rejected rather than silently changing the next capture.

The external generator's state is not owned by the layer. Save it separately if you pass `generator=`. Cross-device, cross-version and cross-platform bitwise equivalence is not promised, consistent with the [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html). Record Torch, source hashes, assets and device information with experiments.

## Patient splits

Split source recordings by patient before generating images. Keep all captures of the same patient in the same partition. Derive validation seeds from recording identity, then render them consistently.

The full ECG Replot trainer has additional patient-split and resume checks. This independent library gives you rendering modules and examples. It does not manage the medical dataset or its permissions.

## Paper-model migration

Version 0.6.0 changes the optical point-spread function and the read/shot noise
combination. It includes their model identifiers in the renderer specification,
so older image-layer checkpoints are rejected by the signature guard. Keep
the earlier package to replay earlier observation pixels.

CLI reports now include `normalized_input`, a SHA-256 fingerprint of decoded
physical signals, per-record sampling rates and valid source lengths. Shapes,
explicit little-endian dtypes and C-order array digests accompany it. This
distinguishes different WFDB waveforms even when their header files match.
The original input-file hash remains available.

Capture metadata includes `canonical_pixels_per_second`,
`canonical_millivolts_per_pixel` and `source_samples_per_canonical_pixel`.
These describe the rectified page before photographic geometry. A 500 Hz
source rendered into fewer than 500 columns per second cannot preserve
every sample's temporal location. The extrema envelope protects narrow
amplitude excursions, not sample chronology inside a single output column.

Version 0.5.0 intentionally changes the sampled paper distribution and renderer signature. An old checkpoint cannot silently resume with the new model. Keep the recorded package version to replay an old experiment, or start an explicitly new rendering experiment. Both `compile_raster` and `compile_scene` settings must match when restoring layer state.

