# Layouts & calibration

Layout names count **rows × columns**. Thus `12x1` prints twelve full-duration rows, while `6x2` prints six rows with two lead windows each. `+r1`, `+r2` and `+r3` add full-duration rhythm strips.

## Pick a layout

```python
from ecg_gpu import ECGImageLayer, Choice

render = ECGImageLayer(parameters={
    "layout": Choice(["3x4", "3x4+r3", "6x2", "12x1"]),
    "rhythm_leads": ["II", "V1", "V5"],
})
```

![Three rows, four columns and three full-duration rhythm strips](assets/gallery/layout-3x4-r3.jpg)

The standard library has {{layout_count}} layouts. Inspect all of them in the [gallery](gallery.html?category=Layouts).

{{layout_table}}

## Sequential and simultaneous windows

With `sequential=True`, each short-row column covers a successive part of the recording. With `sequential=False`, each column starts at time zero. Rhythm strips span the full printed duration in both modes.

Check `metadata['start_s']`, `metadata['end_s']`, `metadata['lead_id']` and `metadata['active']` when supervising waveform recovery. A repeated lead can occur in several printed slots.

## Cabrera ordering

Cabrera layouts print `-aVR`. The output channel retains canonical aVR identity, and `instance_amplitude_sign` is `-1` for the inverted printed occurrence. The source waveform remains in standard lead order.

![Cabrera ordering with the explicitly inverted aVR lead](assets/gallery/layout-3x4-cabrera.jpg)

## Speed, gain and pulses

| Parameter | Index 0 | Index 1 | Index 2 |
| --- | --- | --- | --- |
| `speed_index` | 12.5 mm/s | 25 mm/s | 50 mm/s |
| `gain_index` | 5 mm/mV | 10 mm/mV | 20 mm/mV |

```python
render = ECGImageLayer(parameters={
    "speed_index": 1, "gain_index": 1, "pulse_present": 1.0,
})
```

`pulse_present` controls the 1 mV, 0.2 s calibration mark. Headers and labels can be hidden independently. For classification targets, unknown units are represented when explicit unit-text support is unavailable. This support test measures geometry, not OCR readability.

Long and dense arrangements need an appropriate aspect ratio. Use a tall page for `12x1` and a wide page for `1x12`. The layout does not automatically increase image resolution.
