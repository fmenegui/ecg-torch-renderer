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
        grid_point_config=GridPointConfig(sigma_pixels=.8),
    ),
).to(device)
batch = layer(signals)  # supplied [B,12,T] mV signals
heatmap = batch['grid_points_mask']       # [B,1,H,W]
xy = batch['grid_points_xy']             # [B,N,2], observed x,y
present = batch['grid_points_valid']      # [B,N], bool
ids = batch['grid_points_ids']            # [B,N,2], row,column
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
tensors. Point export uses center camera rays against the actual damaged mesh
and occluders. Canonical grid phase, ppm and header exclusion come from the
rectified render. `return_geometry=True` is optional.

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

Positions locally invert the **same sampled inverse UV map** that projects the
image. A nearest camera-pixel seed is associated with each lattice ID. An
inverse-Jacobian step estimates its subpixel position. Acceptance requires a
round trip within `max_error_pixels`, default .2 canonical pixels, visibility
at least `min_visibility`, default .99, and a bounded nonsingular local solve.

This is an approximate inverse of a sampled map, not exact forward projection
of the physical mesh. The tolerance measures consistency with that map, not
absolute accuracy against an unsampled scene. Independent tests include affine
and pinhole forward projections. Small, grazing or discontinuous regions can
be rejected even when a physical node is partly visible. Failed solves are
explicitly invalid. Hidden positions are not reconstructed, and
`visible_only=False` is rejected for grid-point supervision.

An absent grid, excluded header, missing paper or opaque occlusion produces no
accepted node there. Visibility is geometric. ECG ink, ordinary document text,
glare or blur can obscure a geometrically present intersection. Readability
and detector uncertainty remain separate tasks. Inference cannot use these
ground-truth maps.

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
