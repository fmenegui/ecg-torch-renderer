# Advanced supervision

Version 0.8.1 includes `instance_visible_fraction` whenever instance outputs are
rendered. Its denominator is canonical-page waveform ink before capture. The
numerator uses inverse-Jacobian area quadrature after crop and geometric
visibility. Use `detection_targets(batch, min_visible_fraction=.25)` to exclude
nearly hidden occurrences. This is not optical readability or a fraction of
recoverable temporal samples.

`photo=False` produces a final rectified document with missing substrate
removed from RGB and visible targets. For offline 3D input, explicitly render
`photo=False, as_texture=True` and retain `paper_support` for mesh clipping.

Use `ECGSupervisionLayer` when your network needs an input image and aligned segmentation or detection targets. All requested outputs come from one render and remain on the input device.

## Request just what you need

```python
from ecg_gpu import ECGSupervisionLayer, SupervisionConfig, RenderConfig

synth = ECGSupervisionLayer(
    fs=500,
    config=RenderConfig(height=384, width=512),
    supervision=SupervisionConfig(
        outputs=("lead_masks", "bw", "wb", "lead_boxes", "instance_boxes"),
        binary=False, threshold=.05, visible_only=True,
    ),
).to(device)
batch = synth(signals)
```

The default requests only `lead_masks`. RGB and the visibility mask are always returned. Baseline targets, scene maps and instance masks are not generated unless required by the selected interface.

| Output | Shape | Meaning |
| --- | --- | --- |
| `image` | `[B,3,H,W]` | Photographed RGB, float32 in `[0,1]` |
| `valid` | `[B,1,H,W]` | Paper and explicit visibility support |
| `lead_masks` | `[B,C,H,W]` | Independent lead trace coverage, default C=12 |
| `lead_ids` | `[B,C]` | Canonical lead IDs in output channel order |
| `trace_mask` | `[B,1,H,W]` | Maximum coverage across selected lead channels |
| `bw` | `[B,1,H,W]` | `1 - trace_mask`, black traces on white |
| `wb` | `[B,1,H,W]` | `trace_mask`, white traces on black |
| `lead_boxes` | `[B,C,4]` | One envelope per selected lead, including repetitions |
| `lead_box_valid` | `[B,C]` | Whether that lead box contains any target pixels |
| `instance_masks` | `[B,15,H,W]` | Separate coverage for each printed occurrence |
| `instance_boxes` | `[B,15,4]` | One box for each printed slot |
| `instance_box_valid` | `[B,15]` | Whether that slot has visible target pixels |
| `instance_lead_ids` | `[B,15]` | Canonical identity of each printed slot |
| `instance_active` | `[B,15]` | Whether the layout/slot is enabled before capture |
| `instance_windows_s` | `[B,15,2]` | Printed source window `[start_s,end_s]` |
| `instance_amplitude_sign` | `[B,15]` | Printed polarity, including inverted aVR |
| `name_masks` | `[B,15,H,W]` | Independent glyph coverage per printed lead-name occurrence |
| `name_boxes` | `[B,15,4]` | Visible lead-name glyph boxes, xyxy exclusive edges |
| `name_box_valid` | `[B,15]` | Whether name pixels survive visibility and threshold |
| `name_lead_ids` | `[B,15]` | Canonical identity of each printed name |
| `name_active` | `[B,15]` | Slot and label enabled before capture |
| `name_amplitude_sign` | `[B,15]` | Printed sign, including the minus in Cabrera aVR |

Instance metadata is returned when you request instance masks or boxes. Slots have fixed shapes for batching. Unused or invisible slots have false box-valid flags and boxes filled with `-1`.

## Two grayscale polarities

<div class="figure-pair"><figure><img src="assets/gallery/supervision-bw.jpg" alt="Black traces on white background"><figcaption>BW: white background, black traces.</figcaption></figure><figure><img src="assets/gallery/supervision-wb.jpg" alt="White traces on black background"><figcaption>WB: black background, white traces.</figcaption></figure></div>

These views contain only the chosen leads. Grid, labels, pulses, table texture and capture shadows are excluded. Soft coverage gives anti-aliased edges. Set `binary=True` for masks containing exactly zero and one.

## Independent channels

```python
supervision = SupervisionConfig(
    outputs=("lead_masks", "wb"),
    leads=("V6", "II", "V1"),       # choose and reorder channels
    binary=True,
    threshold=.3,
)
```

The default order is `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`. `leads` can select a subset or reorder channels. It also selects which leads contribute to BW/WB and the union mask. Instance outputs still describe all printed slots.

![Color visualization of twelve independent channels](assets/gallery/supervision-channels.jpg)

Colors visualize the masks. The actual target is a tensor of separate channels. At crossings, more than one channel may be positive. Use sigmoid/BCE or another multilabel objective. A mutually exclusive softmax target loses this information.

## Lead boxes and printed-instance boxes

![Bounding boxes around each separate printed occurrence](assets/gallery/supervision-instance-boxes.jpg)

`lead_boxes` merge all visible occurrences of one canonical lead into one envelope. A long rhythm strip can make that envelope much larger than its short-row box. Use `instance_boxes` when training detection of individual printed traces.

Boxes use **integer pixel-edge xyxy coordinates**. The right and bottom edges are exclusive. For example, `[3,2,7,5]` covers columns 3–6 and rows 2–4. Box extraction uses pixels whose coverage is `> threshold`. The default threshold is `.05` so thin anti-aliased edges can contribute. Increase it if your target definition requires stronger coverage.

## Visibility and camera degradation

`visible_only=True` excludes torn paper, out-of-frame content and explicit opaque occlusion. `valid` can contain soft visibility values. It does not measure OCR readability, contrast or recovery confidence.

`visible_only=False` retains projected traces behind explicit occluders where the projection exists. Out-of-frame pixels still do not exist in an image-space target. This option can support amodal objectives, but it does not make invisible signal segments observable.

Noise, blur and JPEG degradation affect RGB, while target masks remain geometric. They are a clean target under the same capture geometry, not a thresholded copy of the noisy RGB.

## Reuse a richer render

```python
rendered = synth.render_batch(
    signals, stage="all", return_scene=True, return_instances=True,
    return_baselines=False, return_instance_boxes=False,
)
batch = synth.supervise(rendered)       # no second render
```

This lets you retain scene maps or provenance while using the same advanced target format. Match the requested instance outputs by rendering with `return_instances=True`.

## Export targets

```bash
ecg-render --input signal.npz --output ecg.png \
  --targets-dir targets \
  --targets lead_masks,bw,wb,lead_boxes,instance_masks,instance_boxes \
  --binary-targets --target-threshold 0.3
```

`targets.npz` preserves tensor arrays. `boxes.json` includes channel names and the coordinate convention. BW/WB PNGs are lossless grayscale exports. Add `--target-leads II,V1,V5` to choose channel order, or `--projected-targets` for targets behind explicit occluders.

Inside training, use tensors directly. File export, NumPy conversion and JSON serialization belong outside the hot path.


## Printed versus source signal

With `RenderConfig(device_filters=True)` the page shows the recorder-filtered signal: a high-pass (`filter_highpass_hz`, 0.05/0.15/0.5 Hz), a low-pass (`filter_lowpass_hz`, 40/100/150 Hz) and an optional mains notch (`filter_notch_hz`, 0/50/60 Hz). Waveform masks follow the printed trace. `batch.reference` remains the untouched source signal and `batch.metadata["printed_signals"]` holds the printed one, so a digitizer can be trained against either.

## Generated report text

`RenderConfig(report_text=True)` prints generated lines in the header band and a footer in the bottom margin: fictional names, identifiers, dates and device fields, a measurement block and speed, gain and filter settings, in English, Portuguese and Spanish. The printed ventricular rate is estimated from the printed lead II, and the footer agrees with the record's speed, gain and (with `device_filters=True`) filters. No interpretation statements are printed.

```python
from ecg_gpu import ECGRenderer, RenderConfig, decode_report_text

renderer = ECGRenderer(RenderConfig(report_text=True, device_filters=True))
batch = renderer(signals, 500)
mask = batch.targets["report_text"]                       # [B,1,H,W] text ink
lines = decode_report_text(batch.metadata["report_text"])  # transcripts, host side
boxes = batch.metadata["report_text"]["boxes"]            # [B,7,4] xyxy per line
```

## Time and voltage maps

`RenderConfig(coordinate_targets=True)` returns `time_s` and `voltage_mv`, `[B,1,H,W]` maps that give, for every pixel inside a layout cell, the source time and the voltage relative to that cell's baseline, plus `coordinate_valid`. Combined with a waveform mask, they turn digitization into dense regression: on trace pixels, `voltage_mv` matches the source signal at `time_s` within about half a stroke width divided by pixels per mV.

## Lead-name detection

```python
from ecg_gpu import ECGSupervisionLayer, SupervisionConfig, lead_name_detection_targets

names = ECGSupervisionLayer(supervision=SupervisionConfig(
    outputs=("name_masks", "name_boxes")
)).to(signals.device)
batch = names(signals)
targets = lead_name_detection_targets(batch)
```

Names are separate objects for each short lead and rhythm strip. The adapter
uses classes 0..11 in YOLO and 1..12 in torchvision. Canonical aVR remains aVR
when its printed label is `-aVR`, with the sign supplied separately. Missing,
fully hidden and below-threshold labels produce no detections. Targets describe
geometric glyph support, not OCR readability under blur or fading.

For an existing RenderBatch, request `return_name_instances=True` before calling
`supervise`. Names use the same inverse geometry and visibility as the ECG.
Existing RGB and targets are unchanged when enabling these additional targets.


## Page corners

`RenderConfig(page_corners=True)` adds `batch.metadata["page_corners"]`, the observed xy positions of the four sheet corners `[B,4,2]` (top-left, top-right, bottom-right, bottom-left, pixel-center coordinates), with `page_corners_in_frame` and `page_corner_residual`. Corners outside the photograph are reported but marked out of frame. They are the geometric corners of the sheet, so missing-corner damage and occluding objects do not move them.

## Clipping and calibration pulses

With `device_filters=True`, `filter_clip_mv` (sampled for a quarter of records, 1.5–3.5 mV; 0 disables) clips the printed amplitude, as recorders that keep each lead inside its row do. The reference signal is never clipped. `RenderConfig(calibration_target=True)` returns the calibration pulses as `batch.targets["calibration_pulses"]`.

## Barcodes, labels and stamps

`RenderConfig(office_marks=True)` adds patient barcodes, QR-style labels and translucent ink stamps, sized in millimeters. Stamps multiply the page, so traces under them stay visible, as with real ink. `batch.targets["office_marks"]` holds the union of marks. Codes encode random bits and are not decodable.
