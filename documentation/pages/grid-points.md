# Major-grid intersections for point detection

Both backends can export only the intersections of the **5 mm** major grid.
These are material lattice nodes, not paper mesh vertices, 1 mm intersections,
ECG fiducials, or detections from RGB. They enable training a detector for
grid-based rectification, including a Dotter-like first stage. No Dotter model
or published pipeline is reproduced here.

## Online torch supervision

```python
from ecg_gpu import ECGSupervisionLayer, SupervisionConfig, GridPointConfig

layer = ECGSupervisionLayer(
    fs=500,
    supervision=SupervisionConfig(
        outputs=('grid_points',),
        grid_point_config=GridPointConfig(sigma_pixels=1.2, return_offsets=True),
    ),
).to(device)
batch = layer(signals)  # supplied [B,12,T] mV signals
heatmap = batch['grid_points_mask']       # [B,1,H,W]
xy = batch['grid_points_xy']             # [B,N,2], observed x,y
present = batch['grid_points_valid']      # [B,N], bool
ids = batch['grid_points_ids']            # [B,N,2], row,column
offsets = batch['grid_points_offsets']    # [B,2,H,W], dx/dy at supervised anchors
```

Use `outputs=('grid_points_mask',)` to return just the heatmap, image and valid
map. Grid-only supervision avoids dense per-lead trace masks. Combine it with
`lead_masks` or `instance_boxes` for targets from the same observation. All batch
operations stay on the input device. The supplied ECG signal is preserved.
Enabling points consumes no random numbers and changes no RGB or existing labels.

For direct rendering, use
`layer.render_batch(signals, return_grid_points=True)`. Grid outputs are placed
in `RenderBatch.targets`, independently of the legacy `vertices` heatmap.
Use `grid_point_config=GridPointConfig(...)` for custom settings.
On `ECGSupervisionLayer`, direct rendering inherits its supervision grid
configuration. `supervise(rendered)` checks configuration provenance and required
targets, rejecting a differently configured or incomplete grid payload.
Single strings such as `outputs='grid_points'` and `leads='II'` are accepted.
When `return_offsets=True`, include `grid_points` in outputs to receive the
offset targets. Mask-only output with requested offsets is rejected clearly.

## Offline ray-traced capture

```python
from ecg_gpu.pathtrace import pathtrace_capture

# One rectified RenderBatch with stage='all', photo=False,
# as_texture=True and return_instances=True.
capture = pathtrace_capture(canonical, config=capture_config,
                           return_grid_points=True)
heatmap = capture['grid_points_mask']
xy = capture['grid_points_xy']
present = capture['grid_points_valid']
```

The same keys and coordinate conventions apply. Offline outputs remain CPU
tensors. Since v0.11, every lattice node is evaluated directly on the final
UV-parametrized triangle mesh and projected through Mitsuba's actual sensor.
Camera rays verify front-surface visibility and material identity, including
damaged topology and occluders. No inversion of a pixel UV map is needed for
offline point coordinates. `return_geometry=True` remains optional.

## Choose the mask size

```python
# Smooth center targets for a one-channel U-Net.
small = GridPointConfig(sigma_pixels=.6)
larger = GridPointConfig(sigma_pixels=1.2)

# Hard disks with an explicit radius in observed output pixels.
disks = GridPointConfig(mask_shape='disk', radius_pixels=2.5,
                       return_offsets=True)
```

Sigma controls Gaussian spread. Radius controls the disk boundary, with a
5-pixel diameter at radius 2.5. An optional Gaussian `radius_pixels` truncates
its support without changing sigma. Without it, the Gaussian uses the existing
square window of radius ceil(3*sigma). Sigma is positive and at most 4, and an
explicit radius is positive and at most 16. Patch splats are chunked to bound
temporary patch work to at most 2^20 entries per chunk across batch, points
and marker offsets. This is not a bound on output tensors, allocator usage
or total device memory. Very small targets can miss every pixel center.

The center stays fractional and **does not move when mask size changes**.
Masks are evaluated from distances to that fractional center, not to a rounded
center. Reuse already projected points with:

```python
from ecg_gpu import rasterize_grid_points
new_targets = rasterize_grid_points(xy, present, batch['valid'], config=larger)
```

Binary disks alone do not uniquely encode subpixel coordinates: nearby
fractional centers can produce exactly the same pixels. Prefer a continuous
Gaussian or retain the separate offset targets. Very wide masks can also merge
neighboring points. Select width against actual grid spacing at inference.

## Target contract

| Key | Shape | Meaning |
| --- | --- | --- |
| `grid_points_mask` | B,1,H,W | Gaussian point heatmap, clipped to geometric validity |
| `grid_points_xy` | B,N,2 | Observed x,y pixel centers, -1 for invalid slots |
| `grid_points_valid` | B,N | Visible node accepted by the inverse-map solve |
| `grid_points_active` | B,N | Node exists in the printed canonical grid before capture |
| `grid_points_ids` | B,N,2 | Canonical row,column identity |
| `grid_points_canonical_xy` | B,N,2 | Exact canonical x,y pixel-center coordinates |
| `grid_points_mm` | B,N,2 | Material x,y in mm relative to canonical pixel center (0,0) |
| `grid_points_error_px` | B,N | Round-trip error in canonical pixels, -1 for rejected slots |
| `grid_points_error_image_px` | B,N | Numerical solve/camera round-trip residual in observed pixels |
| `grid_points_offsets` | B,2,H,W | Optional dx/dy from a nearest-pixel anchor to the fractional center |
| `grid_points_offset_valid` | B,1,H,W | Optional support for offset regression |
| `grid_points_anchor_xy` | B,N,2 | Optional integer x,y anchor, -1 when unsupported |
| `grid_points_anchor_valid` | B,N | Optional unique visible anchor flags |

Image coordinates start at `(0,0)` at the top-left **pixel center**, unlike
detection-box pixel-edge coordinates. IDs index a row-major padded array and
remain stable across views of the same phased lattice. The printed phase is
reduced modulo 5 mm, so neighboring rows/columns differ by exactly 5 mm in
material coordinates. Perspective and deformation change observed distances.

Default capacity is 128 rows by 128 columns, N=16384. Padding is explicit through
`active` and `valid`. Overflow raises instead of silently dropping points.
Increase `rows`/`columns` for larger physical sheets. Inactive padding positions
are defined but are not training labels.

The heatmap sigma, default .8 output pixels, is a training-target parameter,
not a physical ink dot, printed line width or optical PSF. For a binary point
mask use `SupervisionConfig(binary=True, threshold=.25)` or threshold the
heatmap. Grid lines and 1 mm intersections are excluded.

## Precision and visibility

Online positions invert the **same sampled inverse UV map** that projects the
image. A camera-pixel seed is associated with each lattice ID. Newton steps use
the exact local bilinear interpolant and its Jacobian, with four refinement
steps by default. Acceptance requires a material residual below
`max_error_pixels`, default .01 canonical pixels, an image-space residual below
`max_image_error_pixels`, default .01 observed pixels, and visibility at least
`min_visibility`, default .99. `refinement_steps=0` disables extra refinement.

Offline positions directly project mesh UV nodes, as described above, and
report material and camera-ray verification residuals. Independent tests
include flat and oblique folded-mesh forward projections at tolerances of
.00015 and .0002 output pixels, plus a nonlinear online bilinear reference.

Subpixel **representation**, numerical residuals, absolute label accuracy and
a trained detector's localization error are different quantities. Float32
positions are not proof of physical accuracy. Online interpolation error
against an unsampled surface remains. Offline precision is relative to the
represented mesh and sensor, not a calibrated real camera or real paper.
These tests do not establish real-image detector performance.

Small, grazing or discontinuous regions can be rejected. Failed solves are
explicitly invalid. Hidden positions are not reconstructed, and
`visible_only=False` is rejected for grid-point supervision.

An absent grid, excluded header, missing paper or opaque occlusion produces no
accepted node there. Visibility is geometric. ECG ink, ordinary document text,
glare or blur can obscure a geometrically present intersection. Readability
and detector uncertainty remain separate tasks. Inference cannot use these
ground-truth maps.

## Train and decode a U-Net

Use a full-resolution head. One channel predicts the center heatmap. Two
optional channels regress dx/dy at unique nearest-pixel anchors. This separates
target width from fractional localization. Anchors use nearest-even rounding,
with offsets in [-.5,.5]. When two nodes share an anchor, offset support is
disabled for both rather than averaging incompatible targets. Increase image
resolution if anchor collisions are frequent.

```python
from ecg_gpu import grid_point_loss, decode_grid_points

prediction = unet(batch['image'])  # B,3,H,W
losses = grid_point_loss(prediction[:, :1], prediction[:, 1:], batch)
losses['loss'].backward()

points = decode_grid_points(prediction[:, :1].sigmoid(),
                            prediction[:, 1:], threshold=.3)
xy_pred = points['xy']
valid_pred = points['valid']
```

Offset loss uses only `grid_points_offset_valid`. Heatmap BCE includes the
surrounding table as negative examples. Class weighting is optional and must
be tuned on held-out data. Both masks and offsets need the SAME resolution.
Loss reductions use float32, including half-precision inputs and support maps.
Unsupported offset entries are masked before regression, so NaN/Inf sentinels
outside support have zero contribution and gradient. Supervised offsets must
be finite and within [-.5,.5]. Heatmap and support targets must be finite in
[0,1]. Invalid tensors fail on the input device without copying them to CPU.
For downsampled heads, transform centers with `(xy+.5)/stride-.5` and regenerate
the targets at that resolution, including sigma/radius in output pixels.

For a one-channel network, pass `offsets=None` to the loss and decoder. The
decoder refines each local Gaussian peak with a three-sample log-quadratic fit.
It does not average all grid points into one global soft-argmax. Its subpixel
formula is exact for isolated ideal Gaussian targets away from boundaries,
not arbitrary predicted heatmaps or binary plateaus. Border peaks without
offsets retain integer coordinates along the unsupported axis. NMS radius and
threshold require validation. Outputs include explicit capacity truncation.

```bash
python examples/grid_point_unet.py --steps 3 --sigma 1.2
python examples/grid_point_unet.py --steps 3 --mask-shape disk --radius 2
```

The examples verify finite losses, gradients in all three head channels and
weight updates on artificial ECGs. They do not demonstrate detector accuracy.
In particular a hard-disk target can produce flat or merged peaks, so use the
Gaussian center head for peak decoding unless disk postprocessing is separately
validated. Save float32 NPZ/tensors for training and exact coordinates. Binary
or 8-bit PNG previews do not retain the full subpixel target information.

## Export and inspect

```bash
python examples/grid_points.py --out build/grid-points --offline --spp 128
```

The example exports PNG images, overlays, **point-only binary PNG masks**,
NPZ arrays retaining soft heatmaps and padded point metadata, and CSV rows for
accepted points. A comparison panel, enlarged detail and JSON report are also
written. Encoding and CPU transfer occur in the exporter, outside online
training. Example ECGs are entirely artificial.

Train a one-channel heatmap head, extract local maxima and estimate subpixel
positions. Evaluate point matching precision/recall and localization error,
then test grid ordering and rectification separately. A heatmap alone does not
establish topology, fill missing nodes or implement grid unwarping.

## Reproducible detector benchmark

Install `pip install -e '.[benchmark,pathtrace]'`. The benchmark extra adds
SciPy for offline matching and interpolation, without changing rendering
dependencies. This experiment caches **both** renderer distributions before
training. Here online means images produced by torch, not fresh rendering
inside each optimizer step.

```bash
python examples/prepare_grid_benchmark.py --metadata ptbxl_database.csv --out build/cohort
python -m ecg_gpu.grid_benchmark --signals build/cohort/signals.npz --out build/pilot --ablations
```

Download `ptbxl_database.csv` from the [PTB-XL 1.0.3 release](https://physionet.org/content/ptb-xl/1.0.3/).
The preparation script explicitly downloads only selected 100 Hz records,
checks WFDB initial values and checksums, and converts ADC values to physical
mV. It selects one recording per patient using folds 1–8 for training, 9 for
validation and 10 for testing. It records file hashes, patient IDs and source
IDs. The benchmark accepts other authorized NPZ cohorts with `signals`, `fs`,
`source_ids`, `group_ids`, `splits`, `lead_names` and `units='mV'`. Input tensors
must be finite N,12,T, with standard leads in order. It rejects duplicate source
IDs and patient groups or identical waveform hashes crossing partitions.

Online, offline and mixed baseline runs share source/view keys, model
initialization, crop random sequence, examples seen and optimizer budget. The
mixed regime alternates backends by source/view key. The two rendering recipes
are different distributions, so this is not a causal isolation of ray tracing
from other camera and paper settings. With `--ablations`, additional mixed
models compare sigma .6 and 2, removal of offsets and disks of radius 2.5.
Default baseline sigma is 1.2. All runs use one model seed. Repeat experiments
over seeds and larger cohorts before making a ranking claim.

Thresholds are selected on validation only. Test output includes one-to-one
maximum-cardinality/minimum-distance matching within 1.5 output pixels, TP/FP/FN,
precision/recall/F1, and median/p95 error **among matched points**. The default
benchmark decoder scores every peak using H*W capacity, so its output is not
truncated by an arbitrary node limit. Sparse radius-neighborhood matching
avoids a dense all-pairs/dummy matrix. Empty/undefined metrics are
null, not perfect scores. JSON reports, a source/view manifest, CSV metrics,
model checkpoints and NPZ predictions are exported. Existing nonempty output
directories are refused. `--cache build/pilot` reuses captures after checking
their hashes, source-cohort hash and geometry configuration, while allowing a
different training budget. CPU matching, file I/O and previews are outside the
renderer and loss hot paths.

To recalibrate and evaluate existing weights without any optimizer steps, use
`python -m ecg_gpu.grid_benchmark --rescore build/pilot --out build/rescored`.
Pass the same `--external` manifest to repeat external inference. Capture and
checkpoint provenance are verified. Both training and rescoring reports mark
their completion status, and rescoring records the previous report hash and
training-driver hash separately from the evaluator version. Use the original
capture directory for this operation, not a report-only export.

`--external external.json` loads local original images from an explicit JSON
manifest. Every image requires `id`, `image`, `sha256`, `source_id` and
`group_id`. Annotated images additionally require `points_xy` in original
pixel-center coordinates, a complete `annotation_region=[x0,y0,x1,y1]` and an
`annotation_method`. Images are resized with preserved aspect ratio and padded
on the right/bottom. Scoring is restricted to the annotated region and uses the
same validation-selected threshold. Unannotated images receive predictions
and overlays only. No external image is used for training or threshold tuning.

The helper `examples/annotate_red_grid_scan.py` proposes intersections in an
axis-aligned clean red-grid region for visual review. These are pixel-derived
**silver references**, with uncalibrated annotation uncertainty. They are not
independent expert gold annotations and do not establish subpixel physical
accuracy. Preserve original dataset licenses when sharing any images or
adapted overlays. Source hashes and point-coordinate files can be kept in
research reports without embedding the source images.

The held-out-node rectification diagnostic uses known lattice identities from
matching against synthetic labels. It explicitly reports **oracle
correspondence** and unsupported nodes outside the interpolation hull. The
`rectify_grid_image` primitive accepts supplied ordered correspondences and
returns rectified RGB with support. Neither diagnostic infers grid ordering,
reconstructs occluded cells or validates foldovers. The experiment does not
yet reconstruct waveforms or report SNR. These require a separate trace model,
layout/time-window handling and signal-level evaluation.

Load a saved baseline for inference with:

```python
import torch
from ecg_gpu.grid_benchmark import GridUNet
from ecg_gpu import decode_grid_points

saved = torch.load('build/pilot/baseline-mixed.pt', map_location=device,
                   weights_only=True)
model = GridUNet().to(device)
model.load_state_dict(saved['state_dict'])
model.eval()
with torch.inference_mode():
    heat_logits, delta = model(image_rgb)  # B,3,H,W, RGB in [0,1]
    prediction = decode_grid_points(heat_logits.sigmoid(), delta,
                                   threshold=saved['threshold'])
```

For the no-offset ablation, pass `offsets=None` to decoding. The public decoder
keeps its fixed-capacity API and truncation flags. The benchmark itself uses
full-image capacity to score every peak.
