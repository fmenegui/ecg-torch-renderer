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

## Physical page model

By default the whole record is fitted to the image width, so a 50 mm/s page shows the same seconds as a 25 mm/s page on a longer sheet. `RenderConfig(page_model="physical")` keeps the plotted paper width fixed instead, like a real recorder:

```python
from ecg_gpu import ECGImageLayer, RenderConfig

layer = ECGImageLayer(fs=500, config=RenderConfig(page_model="physical", plot_width_mm=250.),
                      parameters={"speed_index": 2, "trace_width_mm": 0.25})
```

| Control | Meaning |
| --- | --- |
| `plot_width_mm` | Plotted width; 250 mm shows 20, 10 and 5 s at 12.5, 25 and 50 mm/s |
| `trace_width_mm` | Trace and pulse stroke width, sampled in 0.18–0.35 mm |
| `grid_line_mm` | Minor grid line width, sampled in 0.05–0.12 mm; major lines are twice as wide |
| `feed_error` | Relative paper-feed speed error of the trace against the pre-printed grid, sampled in ±1.5 % |

Records shorter than the window end early and later columns stay empty. Pixel `line_width` and `grid_width` overrides are rejected in this mode. `capture_scenario(name, page_model="physical")` and `render_ecg(..., page_model="physical")` select it too.

## Stroke geometry

The default raster measures each pixel's distance only to the curve inside its own pixel column, so steep QRS strokes print thinner than flat segments. `RenderConfig(stroke="isotropic")` also checks `stroke_radius` neighbouring columns (default 3) and keeps the perpendicular width close to `line_width` (or `trace_width_mm` in the physical page model) at every slope. It costs roughly `2 * stroke_radius + 1` times the legacy raster work.

```python
from ecg_gpu import RenderConfig

config = RenderConfig(page_model="physical", stroke="isotropic")
```

## Lead-switch marks

`RenderConfig(lead_marks=True)` prints a short vertical mark on the baseline at each column boundary where the lead changes. `mark_present` (sampled at 85 %) and `mark_height_mm` (1.5–5 mm) control it. Marks appear in the image and in `batch.targets["lead_marks"]`, never in a lead's waveform mask.
