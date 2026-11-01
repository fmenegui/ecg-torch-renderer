# Changelog

## 0.13.0

- Add opt-in per-occurrence lead-name masks, boxes and detection adapters.
  Repeated rhythm labels and inverted Cabrera aVR retain explicit identities.
- Export paired digitization pages, joint waveform/name YOLO detection and
  single-instance segmentation crops with PNG/BMP masks and JSON signals.
- Separate crops containing foreign traces from single-instance crops, and
  count exact crossing pixels independently. Targets keep clean ownership.
- Preserve original mV samples, absolute source times, physical calibration,
  printed polarity, visibility, inverse geometry and file/source hashes.
- Add patient-group train/validation/test splits and staged streaming export
  that checks every capture against a source plan before publishing output.
- Provide ecg-dataset, an artificial integration example and contract tests.
  Task organization acknowledges SynthECG. No upstream code or data is bundled.

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


