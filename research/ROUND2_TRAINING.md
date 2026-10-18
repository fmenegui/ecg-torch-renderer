# Round 2: usable segmentation and detection supervision

## Scope and target definitions

The new adapters train on supplied ECG waveforms, without changing physiological labels. Canonical detector IDs are `I=0`, `II=1`, `III=2`, `aVR=3`, `aVL=4`, `aVF=5`, `V1=6` through `V6=11`. Repeated rhythm strips are separate printed objects with the same lead ID. Cabrera's inverted aVR remains aVR, with inversion represented by the existing amplitude-sign metadata.

Detection boxes tightly enclose **visible waveform ink** above the configured mask threshold. They exclude lead names, calibration pulses, headings, and blank lead-cell areas. This task definition must match real-data annotation. A detector trained on trace extents should not be evaluated against boxes of entire layout cells as though the targets were equivalent. Fully missing, cropped-out, inactive, or occluded traces do not create objects.

`ecg_gpu.training.padded_detection_targets(batch)` retains fixed-size tensors on the input device. Pixel boxes use exclusive right and bottom edges. It clips boxes to the image and provides an explicit validity mask. No host scalar reads or transfers occur. `detection_targets(batch)` compacts valid instances to Ultralytics-style `batch_idx`, `cls` and normalized center/width/height tensors. Dynamic indexing may synchronize CUDA internally. Use the padded adapter when strict fixed-shape execution is required.

`detection_targets(batch, format='torchvision')` returns one detection dictionary per image. Class labels default to **1..12**, because torchvision reserves zero for background. Set the detector to 13 classes including background. A custom zero-background detector can request `label_offset=0`. YOLO always uses canonical 0..11.

No YOLO segmentation export is offered. A sparse, disconnected ECG curve is not safely represented as a single filled contour. U-Net supervision uses independent lead-mask channels, so crossings can belong to more than one lead.

## U-Net integration

```python
from ecg_gpu import ECGSupervisionLayer, RenderConfig
from ecg_gpu.training import masked_segmentation_loss

synth = ECGSupervisionLayer(fs=500, config=RenderConfig(height=384, width=512)).to(device)
batch = synth(signals.to(device))
logits = unet(batch['image'])  # [B,12,H,W]
loss = masked_segmentation_loss(logits, batch['lead_masks'], batch['valid'])
loss.backward()
```

The loss treats lead channels independently. An entirely ignored image has finite zero loss and zero gradients. Optional `pos_weight` accepts one weight per lead, and `channel_valid[B,C]` supports task-specific missing annotations. `batch['valid']` weights observable paper pixels. Table/background pixels are ignored under that support definition. For a segmentation task that should learn table negatives, explicitly supply appropriate support instead. Validity is a geometric observability signal, not a guarantee that a blurred trace is recoverable.

## YOLO dataset and training

```bash
python examples/export_yolo.py --out build/ecg-yolo --groups 4 --views 2
pip install ultralytics
python examples/train_yolo.py --data build/ecg-yolo/dataset.yaml --epochs 1
# Direct actual YOLO optimizer smoke, bypassing the validation backend:
python examples/train_yolo.py --data build/ecg-yolo/dataset.yaml --loss-smoke
```

The exporter writes `images/train`, `images/val`, `labels/train`, `labels/val`, `dataset.yaml` and `manifest.json`. Empty examples receive empty label files. Each line is `class x_center y_center width height` normalized to image size. The manifest preserves source IDs, group IDs, split membership, source hashes when supplied, PNG/label hashes, renderer configuration, seeds, and instance-slot correspondence.

For real datasets, wrap rendered batches with:

```python
from ecg_gpu.dataset_export import samples_from_batch, export_yolo_dataset

records = samples_from_batch(batch,
    source_ids=recording_ids,
    group_ids=patient_ids,
    sample_ids=unique_render_ids,
    provenance=per_image_json_metadata)
export_yolo_dataset(records, 'my-dataset')
```

Split groups are patients when known, otherwise original recordings. All views, crops, and recordings of one patient need the same group ID. The exporter rejects a source ID or supplied source hash under different group IDs, duplicate filenames, malformed image pixels, and output directories containing existing files. It requires two independent groups when validation is requested. This prevents recognized source-level leakage, but cannot infer a patient's identity when callers supply incorrect grouping. The demo's amplitude-scaled artificial signals test integration only and do not represent independent clinical subjects.

The training example constructs a YOLO architecture from YAML with `pretrained=False`, so it does not automatically download weights. It disables reflections, mosaic, mixup and copy/paste because they change page geometry, time direction, or polarity. Ultralytics is optional and its own dependencies and version compatibility remain external to the renderer. `dataset.yaml` contains an absolute root path. Update that field if relocating the exported dataset.

## Validation evidence

- `python -m unittest discover -s tests -p test_training_adapters.py -v`: 7 passed, 1 skipped because CUDA hardware is unavailable.
- Independent analytical coordinate fixtures cover normalized xywh, exclusive-edge area, repeated rhythm strips, crop clipping, missing and nonfinite boxes, visibility-filter opt-in, canonical IDs, and torchvision background offsets.
- Actual PNG decoding and label parsing cover export format, deterministic group membership, repeated source variants, provenance, and rejection before file writes for leakage or unsafe names.
- `python examples/export_yolo.py --out build/round2-yolo-smoke --groups 4 --views 2 --width 192 --height 128`: passed, wrote 8 documents under 4 groups.

- Two actual TinyUNet optimizer steps passed on rendered RGB and 12-channel masks. Loss went from 0.6812198 to 0.6708083 with a verified network weight update.
- Installed optional Ultralytics 8.4.182 and ran `examples/train_yolo.py --loss-smoke` on the exported PNG/txt files. A genuine 12-class YOLO11n model constructed from YAML completed loss, backward, and AdamW update without pretrained weights. Reported components: box loss 6.73219, class loss 5.14089, DFL loss 4.29181.
- Completed a genuine Ultralytics training epoch and validation on CPU with `--epochs 1 --imgsz 128 --batch 2`. Six training documents and two validation documents were read successfully. Training reported box loss 4.435, class loss 3.747, DFL loss 2.709. Validation mAP was zero, as expected for this very short integration run on tiny artificial data. This is not an accuracy result or a trained product.
- The initial full-trainer attempt exposed a torch/torchvision wheel mismatch: torch 2.14.1+cpu with torchvision 0.29.1 lacked `torchvision::nms`. Reinstalling the matching torchvision 0.29.1+cpu wheel from the official PyTorch CPU index with `--no-deps` resolved the dependency failure. No renderer change or test bypass was used.
- Hardware: AMD EPYC 9V74 CPU, Python 3.12.14, torch 2.14.1+cpu, torchvision 0.29.1+cpu. The full trainer log is in `ROUND2_YOLO_SMOKE.log`.

The tests establish software/data-format integration, not photographic realism, clinical generalization, or useful detector accuracy. CUDA throughput and end-to-end CUDA training remain unavailable in this environment.
