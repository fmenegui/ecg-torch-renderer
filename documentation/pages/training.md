# Train an image model

## Paired task datasets

The offline `ecg-dataset` command exports three tasks from the same captures
and patient-group partition. It does not download datasets or train a model.
It accepts local NPY/NPZ and optional WFDB `.hea` inputs through this manifest:

```json
{
  "records": [
    {"input": "record-a.npz", "source_id": "record-a", "group_id": "patient-a"},
    {"input": "record-b.npz", "source_id": "record-b", "group_id": "patient-b"},
    {"input": "record-c.npz", "source_id": "record-c", "group_id": "patient-c"}
  ]
}
```

Paths are relative to the manifest. Each file contains one recording. All
recordings of the same patient must use the same `group_id`. NPZ can declare
`fs`, `units`, `lead_names` and valid padded `lengths`. A manifest entry can
provide `fs` and `units` when appropriate. WFDB units come from its channels.

```bash
ecg-dataset --manifest sources.json --out build/paired --views 3 \
  --layouts 3x4+r1,6x2,12x1 --scenario phone_photo \
  --val-fraction .2 --test-fraction .2

python examples/export_paired_dataset.py --out build/artificial-demo
```

The example uses entirely artificial signals and groups. For supplied signals,
the generator changes document/capture parameters only. `--gain-mm-per-mv 20`
can increase physical overlap without altering the source waveform. The
command first checks source identities, then streams captures one at a time
into temporary staging. Dense target memory does not grow with dataset size.
The final output appears only when every planned sample has succeeded.
Existing nonempty outputs are refused.

| Directory | Contents |
| --- | --- |
| `digitization/{split}/{sample}` | RGB, validity, full original signal in NPZ/JSON, calibration metadata and inverse geometry |
| `detection/images/{split}` | The exact same encoded pages |
| `detection/labels/{split}` | Joint YOLO boxes, class 0 for waveform regions and 1..12 for names I..V6 |
| `segmentation/{split}/with-overlap` | Crops containing another printed waveform instance |
| `segmentation/{split}/without-overlap` | Crops with only the target instance above threshold |
| `manifest.json` | Group assignments, source/file hashes, object slots, overlap counts and omissions |

Each segmentation crop includes RGB, a clean single-instance grayscale
`mask.png`, thresholded binary `mask.bmp`, `valid.png` and `signal.json`.
Repeated occurrences remain separate. The overlap category means that foreign
waveform ink exists anywhere in the crop, including its padding. The manifest
also reports foreign instance slots and exact crossing-pixel counts. Neither
quantity estimates recovery difficulty or optical readability.

JSON references preserve original unresampled mV samples and absolute source
times in the printed half-open interval `[start,end)`. `signal_mv` retains the
canonical input polarity. `printed_signal_mv` applies the explicit printed
sign. A cropped or occluded image does not make every sample observable, and
the full printed window is not silently stretched to the crop width.
`metadata.json` retains the canonical x origin, baseline, pixels/mm, speed and
gain. `geometry.npz` stores the inverse field with `align_corners=False`, so
image pixels can be associated with their original document coordinates.

The Python interface can reuse an existing render:

```python
from ecg_gpu import paired_samples_from_render, export_paired_dataset

rendered = synth.render_batch(signals, stage="trace", return_instances=True,
    return_name_instances=True, return_field=True, return_baselines=False)
records = paired_samples_from_render(rendered,
    source_ids=source_ids, group_ids=patient_ids, sample_ids=sample_ids)
manifest = export_paired_dataset(records, "build/paired")
```

For a lazy iterable, pass `source_plan` with each planned sample's IDs and
`provenance.source_sha256`. All captures must match that plan exactly. The
CLI creates it automatically. Grouping checks cannot identify falsely supplied
patient IDs. Keep all external test patients excluded from training sources.
The YOLO configuration stores the absolute output path. Update that path if
moving the dataset. Standalone `lead_name_detection_targets` uses its own
0..11 canonical vocabulary rather than the joint export's 1..12 name classes.

Task organization follows [SynthECG](https://github.com/rezakarbasi/ecg-image-and-signal-dataset),
Rahimi, Karbasi and Vahabie, [preprint 2025](https://arxiv.org/abs/2506.06315)
and [journal article 2026](https://doi.org/10.4103/jmss.jmss_58_25).
This is an independent implementation. No upstream code or dataset is bundled.
These export checks do not establish digitization accuracy on real images.

Major-grid keypoints are available in both backends. See
[5 mm grid-point supervision](grid-points.html) for point-only masks, identified
coordinates, visibility flags and an online/offline export example.

Instance batches include geometric visible-ink fractions. Both detection
adapters and `export_yolo_dataset` accept `min_visible_fraction`. The grouped
export example exposes `--min-visible-fraction`. Tiny artificial training runs
remain integration checks, not real-image accuracy benchmarks.

Version 0.7.0 includes a complete YOLO detection export/training example and
independent U-Net loss helper. The twelve lead channels are independent waveform
masks, not mutually exclusive classes or filled rectangular regions.

## Detection targets

```python
from ecg_gpu import ECGSupervisionLayer, SupervisionConfig, detection_targets
synth = ECGSupervisionLayer(supervision=SupervisionConfig(
    outputs=('lead_masks', 'lead_boxes', 'instance_boxes'))).to(signals.device)
batch = synth(signals)
yolo_targets = detection_targets(batch)
torchvision_targets = detection_targets(batch, format='torchvision')
```

YOLO uses canonical classes 0..11 and normalized center-x, center-y, width,
height. Torchvision defaults to 1..12 to reserve class zero for background.
Empty and inactive slots are removed. The fixed-shape
`padded_detection_targets` alternative avoids dynamic CUDA indexing.

Use `instance_boxes` to detect each printed occurrence. A short II and repeated
long II strip are two objects with the same class. `lead_boxes` is a semantic
envelope and can span disconnected occurrences. All boxes enclose visible
waveform ink, with exclusive right/bottom pixel edges. They exclude text and
pulses and are not whole layout cells.

```bash
python examples/export_yolo.py --out build/yolo --groups 4 --views 2
python examples/train_yolo.py --data build/yolo/dataset.yaml --epochs 1
```

The optional Ultralytics package requires compatible Torch/torchvision builds.
Install both from the same CPU or CUDA index before installing `.[yolo]`.
The example trains an architecture from YAML without pretrained weights.
Actual one-epoch training/validation was executed on artificial data. A zero
mAP on the tiny validation set is recorded, so this is integration evidence,
not detector accuracy evidence.

The exporter writes PNGs, YOLO text labels, `dataset.yaml` and image/label/source
provenance. It groups recordings before splitting and rejects known source/hash
leakage. Supply actual patient IDs as group IDs. Multiple recordings from one
patient must share a group. Source identity is caller-supplied and cannot prove
absence of undeclared patient overlap. Disconnected waveform masks are not
converted into misleading YOLO segmentation polygons.

## Optional offline 3D data

Install `.[pathtrace]` and run `examples/pathtraced_capture.py`. This generates
actual Mitsuba light transport with the same camera rays used for visible
front-surface masks and boxes. It exports NPZ arrays and YOLO detection labels.
All views of its single source deliberately stay in one partition. This route
uses CPU/offline file work and is separate from the torch training hot path.

The meshes preserve discrete material edge lengths and permit curl/fold views.
They are prescribed developable shapes, not cloth dynamics. Pixel coverage uses
stratified ray samples and is separate from optical readability.

Load signals, move them to the training device, render there, then pass RGB to your network. DataLoader workers can read and preprocess source files. Keep CUDA rendering in the training process.

## U-Net segmentation

```python
import torch
from torch.nn import functional as F
from ecg_gpu import ECGSupervisionLayer, RenderConfig

synth = ECGSupervisionLayer(
    fs=500, config=RenderConfig(height=384, width=512)
).to(device)
optimizer = torch.optim.AdamW(unet.parameters(), lr=3e-4)

for signals in signal_loader:
    signals = signals.to(device, non_blocking=True)
    batch = synth(signals)
    optimizer.zero_grad(set_to_none=True)
    logits = unet(batch["image"])       # [B,12,H,W]
    loss_map = F.binary_cross_entropy_with_logits(
        logits, batch["lead_masks"], reduction="none"
    )
    support = batch["valid"]
    loss = (loss_map * support).sum() / (
        support.sum() * logits.shape[1]
    ).clamp_min(1)
    loss.backward()
    optimizer.step()
```

Every lead is an independent output channel. Soft targets preserve fractional sheet-edge coverage exactly once. The full `valid` map supplies loss support, while visibility given a sheet hit gates the projected masks. Keep logits unbounded during training and use `.sigmoid()` for predicted probabilities at evaluation. Trace pixels are sparse, so tune class weighting or an additional Dice objective against validation data. A finite loss and a successful backward step do not establish good digitization performance.

## Run the included U-Net example

```bash
python examples/segmentation.py --steps 3
python examples/segmentation.py --device cuda --compile-raster --compile-scene --steps 20
```

The complete example defines a small encoder, bottleneck, upsampling path and skip connection. It verifies that a 12-channel output receives gradients and that optimization changes network weights. It uses artificial waveforms.

## Classification with nn.Sequential

```python
from torch import nn
from ecg_gpu import ECGImageLayer

pipeline = nn.Sequential(
    ECGImageLayer(fs=500, config=RenderConfig(height=384, width=512)),
    image_network,
).to(device)
logits = pipeline(signals)
```

Use the RGB-only layer when your model does not need masks. It avoids allocating dense supervision targets.

## Mixed precision

```python
synth = synth.to(device)              # preserve float32 renderer buffers
with torch.autocast(device_type="cuda", dtype=torch.float16):
    batch = synth(signals)
    logits = unet(batch["image"])
    loss_map = F.binary_cross_entropy_with_logits(logits, batch["lead_masks"], reduction="none")
```

The renderer disables autocast internally and returns float32 geometry and pixels. The network can still use AMP. In a full fp16 training loop, use `torch.amp.GradScaler("cuda")` for backward and optimizer stepping. Do not call `.half()` on the renderer or on a Sequential pipeline containing it.

## Detection and waveform recovery

Use instance boxes and IDs to train a detector. Use `instance_windows_s` and `instance_amplitude_sign` to associate detected printed windows with source samples. Aggregate per-lead masks only when repeated occurrences should share one output channel.

The reference signal is not automatically a waveform-regression target for every crop. Supervise only the source windows that your image/task definition makes observable.

## Distributed training

Create one synthesizer per rank, on that rank's device. Give each rank a distinct seed, for example `base_seed + 1009 * rank`. Save renderer state alongside model and optimizer state. Restore per-rank randomization counters when exact continuation matters.

DDP can wrap the trainable image network. The synthesizer has buffers and no trainable parameters. For data loading, use pinned memory and nonblocking transfers when appropriate. Measure the full training step before selecting renderer batch size or resolution.

