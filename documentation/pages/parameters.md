# Configure & randomize

In 0.8.1, `angle` is absolute radians. Overriding only `quarter` recomposes
rotation while retaining sampled angular jitter. Overriding only `angle`
derives its nearest quarter-turn class. Contradictory explicit pairs raise.
`parameters['photo']=0` gates photographic geometry for selected records.
The formal `photo=False` selects a final rectified document.

`lead_present` and `photo` accept only zero/one. Use a Bernoulli distribution
for random presence rather than passing a fractional probability as a fixed
presence value. Large custom fonts must fit the static `max_font_size` bound,
including the header's 1.15 size multiplier.

Use `RenderConfig` for structural cost and capture defaults. Use `parameters` for individual physical and appearance controls. Unspecified parameters use the built-in random sampler.

## Fixed values and distributions

```python
from ecg_gpu import ECGImageLayer, RenderConfig, Uniform, Choice, Bernoulli

render = ECGImageLayer(
    config=RenderConfig(height=384, width=512, profile="photo", severity=.5),
    parameters={
        "layout": Choice(["3x4+r1", "6x2", "12x1"]),
        "angle": Uniform(-.18, .18),
        "zoom": Uniform(.8, 1.05),
        "header_present": Bernoulli(.9),
        "grid_present": Bernoulli(.95),
        "paper_color": [1.0, .98, .96],
        "background_type": "wood",
    },
)
```

Fixed values and sampler constants become registered buffers. `.to(device)` moves them once. `Uniform`, `Choice` and `Bernoulli` sample whole batches using Torch on that device.

| Form | Best use |
| --- | --- |
| Scalar | Fix one value across the batch |
| Vector matching parameter shape | Fix colors, directions or slot controls |
| `Uniform(low, high)` | Sample continuous quantities |
| `Choice(values)` | Sample named layouts, materials or discrete values |
| `Bernoulli(p)` | Sample presence flags |
| Per-call device tensor | Control each sample separately |

Do not use `Uniform` for categorical IDs. Named categories are resolved at construction. The [API reference](api.html#rendering-parameters) lists every parameter and its actual tensor shape.

## Override one batch

```python
batch_size = signals.shape[0]
angles = torch.linspace(-.1, .1, batch_size, device=signals.device)
image = render(signals, parameters={"angle": angles})
```

Per-call overrides take precedence over fixed values and samplers. Repeated Python/CPU overrides can introduce transfers. For steady-state training, configure constants at construction and pass per-sample overrides as device tensors.

## Curriculum

Start with readable pages, then widen the augmentation distribution. Track performance separately for crops, surfaces, shadows and compression. Changing image difficulty without measuring waveform accuracy can hide a poorly calibrated model.

```python
# No CPU tensor is created for this batch-specific override.
strength = torch.full((signals.shape[0],), .3, device=signals.device)
image = render(signals, parameters={"surface_strength": strength})
```

## CLI distributions

```bash
ecg-render --input signal.npz --output images \
  --parameters '{"layout":{"choice":["3x4+r3","6x2"]},"angle":{"uniform":[-0.15,0.15]},"grid_present":{"bernoulli":0.9}}'
```

JSON supports fixed values and the `uniform`, `choice` and `bernoulli` distribution forms.

