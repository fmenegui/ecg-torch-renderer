# Changelog

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

