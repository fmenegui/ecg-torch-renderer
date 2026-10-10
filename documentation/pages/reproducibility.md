# Reproducibility

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

The external generator's state is not owned by the layer. Save it separately if you pass `generator=`. Cross-device, cross-version and cross-platform bitwise equivalence is not promised. Record Torch, source hashes, assets and device information with experiments.

## Patient splits

Split source recordings by patient before generating images. Keep all captures of the same patient in the same partition. Derive validation seeds from recording identity, then render them consistently.

The full ECG Replot trainer has additional patient-split and resume checks. This independent library gives you rendering modules and examples. It does not manage the medical dataset or its permissions.
