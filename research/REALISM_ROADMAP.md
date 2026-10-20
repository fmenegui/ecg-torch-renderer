# Photorealism roadmap

Proposal after the 2026-10-20 correctness audit. This document proposes work.
The proposal is retained as the research target. Version 0.9 implements bounded
mechanisms for all eight areas, documented in `documentation/pages/realism.md`.
Measured device templates/material presets, a full thin-shell collision solver,
calibrated chemistry/ISP and the real-image acceptance evidence remain open.
No real-image results are claimed.

## Objective

Generate photographs and scans that preserve known signal/document geometry
while matching defined acquisition domains. Visual credibility and training
utility are separate outcomes. A dramatic image is not necessarily a realistic
clinical capture. We should target ordinary convincing images first, then rare
severe failures, and retain the supplied physiological waveform exactly.

## Why the current images still look synthetic

The release gallery repeats generic headers, uniform printing and a small set
of scene appearances. The fast crumpled sheet has shallow relief. Thin cuts can
look like pen strokes, and hole boundaries lack thickness or exposed fibers.
Coffee deposits vary but still have a recognizable closed contour. These are
qualitative observations of the release and audit galleries, not a blinded
real-versus-synthetic experiment.

The 0.8.1 corrections improve correctness and demo diversity. They do not solve
paper mechanics, calibrated sensor response or photographic equivalence.

## Proposed work in priority order

| Priority | Component | Concrete mechanism | Acceptance evidence |
| --- | --- | --- | --- |
| 1 | Device document templates | Distinct header blocks, font metrics, margins, grid ink, label placement, feed boundaries and calibration conventions. Use fictional patient/header data. Keep signal time and voltage scale explicit. | Overlay templates against authorized blank/reference prints. Check dimensions in millimeters and lead identity. Hold out template families in evaluation. |
| 2 | Paper and thermal printing | Measured paper albedo and microstructure, print-head streaks, local retention, correlated ink variations and paper-feed patterns. Separate substrate, grid, waveform and marks. | Compare blank and printed scans, line widths, spatial spectra and local contrast at final output resolution. Do not tune from the final test set. |
| 3 | 3D paper with contact | A thin-sheet model with bending resistance, persistent crease history, table contact and collision handling. Preserve material UVs. Precompute geometry/visibility banks for the fast training path. | Measure triangle strain, penetration, self-intersections and target/ray alignment. Compare a folded sheet photographed before and after opening. Do not equate generic cloth parameters with calibrated paper mechanics. |
| 4 | Torn and cut edges | Explicit finite thickness, lifted cut lips, irregular fracture paths, exposed fiber scale and damaged-coating appearance. Damage changes shared geometry, not a painted black mask. | Close-up references at matched magnification. Primary-ray labels stop at missing substrate. Shadows and edge scattering change with actual geometry and light direction. |
| 5 | Camera observation | Depth-dependent defocus, coherent illumination and reflections, exposure/noise calibration, lens distortion and optional measured ISP approximation. Treat JPEG as a final observation effect. | Photograph the same target with controlled angle, illumination, distance and camera. Fit on development captures and evaluate held-out conditions/devices. |
| 6 | Monitor capture | An emissive display plane, pixel/subpixel aperture, sampling between panel and camera, surface reflections and optional temporal refresh/rolling-shutter interaction. | Capture the same known ECG screen image at several distances, scales and refresh conditions. Explain spatial moire and temporal bands with their actual causes. Keep the screen-to-signal map known. |
| 7 | Liquid and storage history | Liquid contact, drying, pigment deposition, substrate warping and thermal-coating response sampled as a coherent history. | Match measured footprint/contrast statistics and their relation to paper deformation. Avoid using stronger stain opacity as a substitute for geometry. |
| 8 | Scene diversity | Authorized image/reflectance banks, multiple table materials, printed sheets, folders and physical occluders with contact and cast shadows. | Split asset families between training and evaluation. Remove trivial synthetic cues such as one fixed tabletop or one generic header. |

## Two execution paths

Keep the Torch renderer as the efficient batched path for moderate open-sheet
deformations. Use physically richer offline captures and shared UV/depth/normal/
visibility banks for severe folds and self-overlap. This makes it possible to
pay expensive geometry/light transport costs once while retaining on-device
training. Reusing a bank requires testing whether bank diversity is sufficient.

Do not use image-to-image diffusion as the default realism layer for paired
training targets. It may modify waveform ink or labels without a known pixel
mapping. Such a layer would need independent content-preservation evidence and
an explicit rejection criterion before its output could be treated as paired GT.

## Controlled reference set

Proposed pilot, not yet collected:

1. Six document templates printed from known signals. Include different lead
   layouts and ordinary acquisition variation rather than only dramatic damage.
2. Two phones, one flatbed scanner and two displays. Capture a balanced subset
   under controlled scale, angle and illumination rather than a full expensive
   Cartesian product.
3. Start with 200 to 300 reference captures. Use blank charts, printed charts
   and fiduciary targets to identify paper, print and camera effects separately.
4. Assign source/patient, template, device and scene IDs before splitting. No
   views of the same patient/source cross a held-out split. Keep a device or
   template family outside fitting when testing transfer.
5. Retain original signals, image hashes, capture settings, calibration target
   geometry and acquisition consent/licensing where applicable.

The pilot count is an engineering starting point. It is not a power calculation
or a sufficient sample size for clinical claims.

## Experiments that decide whether a component helps

- Train the same detector/digitizer with equal budgets on clean prints,
  conventional augmentation, the current renderer and each new component.
- Report instance AP by lead, fine-trace segmentation, complete failures and
  waveform reconstruction error/SNR. Separate readable from information-lost
  regions. Group uncertainty estimates by independent patient/source.
- Run ablations for template, paper, geometry and camera. A visual improvement
  that worsens held-out signal reconstruction is not automatically adopted.
- Run a blinded visual comparison at fixed output sizes. Report participant
  count, confidence intervals and results by capture type. Control background,
  header and resolution clues before attributing detection to physical realism.
- Compare with ECG-Image-Kit under matched signals, splits and output resolution.
  Do not infer superiority from the number of effects or a selected gallery.

## External technical basis

- [ECG-Image-Kit paper](https://arxiv.org/abs/2307.01946): existing primary work
  on paired synthetic ECG images and document artifacts. It is a baseline to
  compare, not evidence that this renderer already generalizes.
- [PhysioNet Challenge 2024](https://moody-challenge.physionet.org/2024/): task
  context and ECG-image digitization evaluation. Follow the actual data access
  and licensing conditions for any benchmark.
- [PBRT, Cameras and Film](https://pbr-book.org/4ed/Cameras_and_Film): image
  formation includes lens effects and sensor measurement beyond pinhole geometry.
- [Mitsuba BSDF reference](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_bsdfs.html):
  available material models. Choosing a BSDF does not establish measured paper
  reflectance or make a material photorealistic by itself.
- [Mitsuba variants](https://mitsuba.readthedocs.io/en/latest/src/key_topics/variants.html):
  LLVM/CUDA execution and differentiated rendering. This enables future inverse
  fitting but does not validate the present sensor or material parameters.
- [DiffCloth](https://arxiv.org/abs/2106.05306): primary research on differentiable
  simulation with frictional contact. Its cloth model requires adaptation and
  calibration before being used to make claims about paper.

The roadmap is our engineering proposal derived from observed gaps and these
references. It does not establish that any proposed mechanism is novel.
