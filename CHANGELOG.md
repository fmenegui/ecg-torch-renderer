# Changelog

## 0.19.0

- Add `RenderConfig(report_text=True)`, opt-in generated report text in the
  header band and bottom margin: up to six header and measurement lines in
  English, Portuguese and Spanish (fictional names and identifiers, dates,
  rooms, devices, intervals) plus a footer with speed, gain and filter
  settings. Text is composed per character from a glyph atlas, with digit
  slots filled on device.
- The ventricular rate printed in measurement lines is estimated from the
  printed lead II by autocorrelation (`ventricular_rate`). Footers agree with
  the record's speed and gain and, with `device_filters=True`, its filters.
  No interpretation statements are printed.
- Targets: `report_text` ink mask. Metadata: per-line glyph ids, presence,
  boxes and rate; `decode_report_text(meta)` returns transcripts for OCR.
- Not combinable with a `document_template` (both use the header band).

## 0.18.0

- Add `RenderConfig(device_filters=True)`, opt-in: the image prints the
  recorder-filtered signal, as electrocardiographs do. Per-record
  `filter_highpass_hz` (0.05/0.15/0.5), `filter_lowpass_hz` (40/100/150) and
  `filter_notch_hz` (none/50/60) are zero-phase magnitude responses applied
  after even-symmetric extension. Waveform targets follow the printed trace;
  `RenderBatch.reference` stays the untouched source and
  `metadata["printed_signals"]` holds the printed version.
- New module `ecg_gpu.device_filters`.

## 0.17.0

- Add `RenderConfig(scan_modes=True)`, opt-in flatbed scanner observation:
  color, grayscale or bilevel (`scan_mode` 0/1/2; 50/30/20 % for records
  without photographic capture) at a scanner resolution `scan_dpi`
  (log-uniform 72-600 dpi) with bilevel cut `scan_threshold`. Low dpi
  integrates and holds each scanner pixel. Targets are unchanged.
- New module `ecg_gpu.scan_modes` with `scan_observation` and
  `scan_resolution`.

## 0.16.0

- Add `RenderConfig(lead_marks=True)`, opt-in: short vertical marks on the
  baseline at every column boundary where the printed lead changes, as most
  recorders print. Controls `mark_present` (85 % by default) and
  `mark_height_mm` (1.5-5 mm). Marks use the trace stroke, appear in the
  image and in a separate `lead_marks` target, and never enter waveform masks.
- Legacy random stream and specification are unchanged when disabled.

## 0.15.0

- Add `RenderConfig(stroke="isotropic")`, opt-in. Each pixel measures its
  distance to the curve segments of `stroke_radius` (default 3) neighbouring
  columns on each side, so steep QRS strokes keep the requested width. With a
  3 px line, perpendicular width at slope 10 is 3.00 px instead of 1.27 px in
  the legacy kernel. The extremum envelope is drawn isotropically too.
- The legacy kernel, its specification and the frozen v0.2.1 fixture are
  unchanged. `research/stroke-width/probe.py` documents the legacy behaviour.

## 0.14.0

- Add `RenderConfig(page_model="physical")`, opt-in. The plotted paper width
  is fixed (`plot_width_mm`, default 250 mm), so paper speed selects the shown
  time window: 20, 10 and 5 s at 12.5, 25 and 50 mm/s. Records shorter than
  the window end early and later columns stay empty.
- In the physical page model, trace and grid widths are millimeters
  (`trace_width_mm`, `grid_line_mm`; major grid lines twice the minor width),
  so physical stroke width no longer changes with resolution, duration or
  speed. Pixel `line_width`/`grid_width` overrides are rejected there.
- Add a paper-feed speed error (`feed_error`, default ±1.5 %) that scales the
  printed trace and calibration pulse against the pre-printed grid. Metadata
  reports the actual pixels per second.
- `capture_scenario(..., page_model=...)` and `render_ecg(..., page_model=...)`
  accept the new model.
- The legacy model is unchanged: same random stream, same specification and
  saved-state signature, same frozen v0.2.1 raster fixture.
- Add `research/STATE_OF_THE_ART_ROADMAP.md` with acceptance criteria.

## 0.13.0

- Add exploration and data-preparation helpers in `ecg_gpu.convenience`,
  exported from the package: `prepare_signals` (arrays in any lead order and
  mV/uV/V to a `[B,12,T]` mV tensor), `render_ecg` (one-call rendering with an
  optional capture scenario), `to_pil` and `describe_parameters`. `LEADS` is
  exported too.
- Error messages for unknown parameters, layouts, scenarios, surface and
  background names and lead names now suggest the closest valid names.
  Signal shape, dtype and device errors report what was received and how to
  fix it. Non-tensor signals raise `TypeError` instead of `AttributeError`.
  Original message prefixes are unchanged.
- `rhythm_leads` accepts lead names in any letter case.
- `demo_signals` gains optional `heart_rate`, `hrv`, `baseline_wander_mv`,
  `noise_mv` and `seed`. The defaults reproduce the previous output exactly,
  and limb-lead identities hold with every option.
- Record a visual realism assessment of 0.12.0 in
  `research/REALISM_ASSESSMENT_V0120.md`.

## 0.12.0

- Add a reproducible grid-detector benchmark (`ecg-grid-benchmark`): patient-
  disjoint PTB-XL sources, equal-budget online/offline/mixed training, target
  ablations, all-peak decoding, sparse exact matching, validated external
  manifests, deterministic cache replay and checkpoint rescoring.
- Record a measured exploratory pilot in `research/grid-benchmark-v0120`:
  24 patients, 96 captures, seven models, one training seed and four test
  patients. It does not establish renderer realism, clinical generalization or
  a CUDA result.

## 0.11.1

- Fix float16 zero-loss and support-reduction overflow in grid-point losses.
- Mask inactive nonfinite offsets before regression and enforce grid target
  provenance.
- Inherit configured defaults, preserve compatible lead-only replay and bound
  temporary splat patches across batch and points.

## 0.11.0

- Separate grid-point mask width from fractional center coordinates, with
  Gaussian and disk controls.
- Add collision-aware offset targets, `grid_point_loss`, a local subpixel
  decoder (`decode_grid_points`) and a full-resolution U-Net example.
- Project offline points directly from mesh to sensor. Refine online
  coordinates with bilinear Newton steps and image-space residual checks.

## 0.10.0

- Export identified 5 mm grid intersections from both the torch and offline
  backends: padded row/column identities, canonical and millimeter positions,
  observed subpixel coordinates and a point-only mask.
- Reject points through explicit inverse-map residual and visibility checks.
  Add analytical identity, affine and pinhole tests and occlusion checks.

## 0.9.1

- Place template unit text and title clearance in physical millimeters,
  fixing overlapping headers. A 72-case typography regression reproduced 54
  failures before the correction.

## 0.9.0

- Add physical document templates, thermal printing, scaled paper-scan
  imports, thin-lens optics, an emissive RGB display model and coupled
  liquid/storage material history.
- Offline sheets gain compliant relaxation, finite thickness, cut lips, fibers
  and reusable primary-ray geometry banks.
- Keep the signal reference immutable and target visibility shared across the
  new models. Measured presets and external photographic equivalence remain
  unverified.

## 0.8.1

- Recompose quarter-turn overrides and derive orientation from absolute angles.
  Reject contradictory explicit orientation/angle pairs.
- Integrate projected unit support using the inverse Jacobian so perspective
  magnification is not mistaken for missing calibration text.
- Apply missing substrate to final rectified documents. Add explicit
  `photo=False, as_texture=True` for intact offline-3D albedo with separate support.
- Make the per-record photo parameter gate geometry and add a static photo
  option to ECGImageLayer and the CLI.
- Size font atlases and static glyph bounds from actual text and a configurable
  font-size limit. Reject oversized overrides rather than silently truncating.
- Supply instance-visible ink fractions to training adapters and grouped YOLO
  export, with documented geometric rather than readability semantics.
- Add checkpointed independent RNG streams and stable source/view seed derivation.
- Add direct/sensor camera specializations and a static JPEG opt-out, preserving
  pixel/RNG parity within their supported domains. Recipes select these paths.
- Validate integer configuration controls and binary lead/photo presence early.
- Expose the offline Mitsuba variant and avoid changing global Dr.Jit thread count.
- Give artificial demo leads distinct morphologies and consistent limb identities.
  This remains an integration example, not a physiological simulator.
- Add audit regressions, a status ledger and a source-backed photorealism roadmap.

## 0.8.0

- Branched, finite and asymmetric creases, irregular accordion spacing and
  oriented micro-wrinkles replace uniformly centered relief patterns.
- Linear-light paper absorption separates oxidation, clustered foxing,
  irregular liquid deposition, fibers, thermal fading and coating darkening.
- Material-aligned tears, edge wear, holes, open slits and missing corners
  share the RGB, target and shadow support. Rich and edge-only cost modes are
  explicit. Geometric waveform ownership survives stains and fading.
- Sixteen coherent capture scenarios include liquid damage, crumpling,
  fragments, cuts, thermal damage and stronger projective capture.
- Offline 3D capture adds seeded irregular sheets, real removed topology,
  paper-support import, physical camera roll and auditable mesh metrics.
- Independent physical/target regressions, native-resolution visual review
  and third-round research reports accompany reproducible galleries.

## 0.7.0

- Audit all 30 layouts with independent per-lead/per-instance mask and box
  oracles, including repeated rhythm strips, Cabrera, crop, tears and marks.
- Add on-device YOLO/torchvision target adapters and masked segmentation loss.
- Export grouped PNG/YOLO datasets with source/provenance hashes and guards
  against known source leakage. Verify actual U-Net and YOLO training steps.
- Move optical integration, exposure, scene resampling and compositing into
  linear light. Preserve HDR energy until sensor saturation.
- Forward-project caster-height shadows, remove detached contact halos,
  use coherent lighting and eliminate strongly periodic synthetic wood.
- Attach pen marks and artifact text to sheet coordinates before deformation.
- Add explicit measured RGB bank loading and an attributed CC0 wood texture.
- Add optional offline Mitsuba 3D light transport with developable unit-edge
  paper geometry, distinct unprinted backside, real camera ray visibility,
  occluders, 12 lead masks, occurrence boxes and complete capture provenance.
- Retain six independent reviews, actual renderer examples and training logs.
- Update fingerprints for the changed camera/scene distribution. Earlier
  checkpoints require the earlier package for observation-pixel replay.

## 0.6.0

- Add seven coherent named capture scenarios, from flatbed scans to low-light
  photographs, with reproducible multi-layout gallery export.
- Integrate finite uniform-shutter motion with Gaussian defocus and combine
  independent read/shot noise variances correctly. Capture appearance changes,
  so image seeds from earlier releases are not visually interchangeable.
- Reject fractional categorical IDs and lengths before conversion, invalid
  opacity/presence values and input overflow in float32 raster coordinates.
- Freeze caller-supplied layout sequences to keep tables and provenance aligned.
- Reduce supervision postprocessing allocations for union and instance-only
  requests, retaining bitwise output equality.
- Export canonical resolution limits and canonical physical-input fingerprints.
- Add independent fidelity, optical, adversarial and allocation tests, primary
  research review and a professional multi-agent acceptance workflow.
- Keep frozen raster fixtures, record CPU evidence and leave CUDA and external
  real-image equivalence explicitly unverified for this release.

## 0.5.0

- Replace elastic-looking default camera warps with material-coordinate bends,
  diagonal corner hinges and finite signed crease facets.
- Couple crease appearance to geometry, normalize matte illumination and remove
  false flat-paper self-shadows at grazing light angles.
- Use nonperiodic multiscale table textures, subtle paper grain, local-height
  cast shadows, controlled shadow softness and fractional sheet-edge coverage.
- Add a coarse-to-fine inverse solve with two pixel refinements, retain a dense
  reference and skip full-resolution normals/strain when not requested.
- Batch three-light shadow rays in bounded chunks and add opt-in `compile_scene`
  for deterministic geometry, lighting and Newton updates with unchanged RNG draws.
- Fix external-bank visibility padding and illumination at differing resolutions.
- Count only actual CUDA kernel trace events, retaining device/annotation metadata.
- Keep exact same-runtime 0.2.1 raster regression, add float32 fixture tolerance
  with useful failure diagnostics, and add independent paper geometry tests.
- Preserve uploaded T4 0.4.0 evidence, regenerate the 125-figure gallery and update
  the GPU Colab notebook to compare current backends.
- Version the paper contract and reject incompatible checkpoint continuation.

## 0.4.0

- Add `ECGSupervisionLayer` and `SupervisionConfig` for aligned RGB, lead masks,
  union/BW/WB views, per-lead boxes and per-instance masks/boxes.
- Support reordered/subset channels, soft/binary coverage and visibility.
- Skip baseline generation and legacy instance-box work in the advanced path.
- Export aligned NPZ targets, pixel-edge JSON boxes and lossless BW/WB PNGs.
- Add a complete small U-Net example and supervision integration tests.
- Include supervision configuration in detailed profiling reports.
- Replace library docs with an English guide, local search, generated API
  reference, responsive themes and 125 actual generated figure recipes.
- Extend the English Colab notebook with U-Net and supervision profiling.

## 0.3.0

- Add optional compiled raster kernels and batched deterministic coverage.
- Preserve extrema and physical layout semantics against frozen 0.2.1.
- Add operator/kernel/memory profiling with separate warmed timing.
- Package a standalone library, console tools and GPU Colab notebook.

## 0.2.1

Frozen numerical/performance reference, retained as the immutable archive
`references/ecg-gpu-generator-v021.zip`.


