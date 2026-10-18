# Round 2: document rendering and physical realism research

Reviewed 2026-10-18. Research role, primary author repositories and publications.
This is a source-backed design review. External datasets were not downloaded,
no comparator was executed, and no photographic-equivalence result is claimed.
The implementation changes in other round 2 reports must be assessed separately.

## Decision

The strongest next step is a hybrid renderer: calibrated torch document creation
and supervised projection, with optional banks of measured or physically rendered
paper geometry, shading and real background assets. Keep expensive acquisition,
asset conversion, path tracing and licensing checks outside training. Replace
periodic procedural cues with measured distributions, rather than merely adding
stronger random damage. No published toolbox establishes a universal 100% realism
guarantee, and this repository has not established one either.

## Relevant primary implementations

| Project | Verified technique | License observed | Useful adoption and limitation |
| --- | --- | --- | --- |
| [Augraphy](https://github.com/sparkfish/augraphy) | Distinct ink, paper and merged-document phases. InkBleed, InkMottling, dirty rollers, folding, shadow and photocopy effects. Spatial augmentation support is documented separately from pixel-only changes. | MIT in author repository | Best offline degradation comparator. OpenCV/NumPy-oriented CPU pipeline is unsuitable as a direct torch training hot path. Port selected ideas and independently validate annotation semantics. Its default office-printer distribution is not an ECG thermal printer calibration. |
| [DocCreator](https://github.com/DocCreator/DocCreator) and [author project](https://doc-creator.labri.fr/) | Ink degradation, bleed-through, adaptive blur, holes and paper deformation, with ground truth. C++/Qt/OpenCV framework. | LGPL-3.0 in repository | Useful offline artifact reference and material-specific damage taxonomy. Do not blindly copy implementation into an MIT package or treat book-page bleed-through as a common ECG default. |
| [Doc3D](https://github.com/cvlab-stonybrook/doc3D-dataset) / [DewarpNet paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Das_DewarpNet_Single-Image_Document_Unwarping_With_Stacked_3D_and_2D_Regression_ICCV_2019_paper.html) | Hybrid synthetic data for paper unwarping, geometric maps and deformable document scenes. | MIT in Doc3D repository [LICENSE](https://github.com/cvlab-stonybrook/doc3D-dataset/blob/master/LICENSE) | Candidate geometry reference. Repository software license alone does not establish the provenance of every included texture or external asset. |
| [DocUNet author benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) | Real document photos, centered crops and flatbed scans, with MS-SSIM and local distortion evaluation. | Dataset redistribution license not resolved on the inspected project page | Good independent geometric benchmark, not an ECG lead annotation dataset. Do not silently bundle it. |
| [UVDoc-Dataset](https://github.com/tanguymagne/UVDoc-Dataset), [paper](https://arxiv.org/abs/2302.02887) | Captured deformed white-paper geometry, replaceable document textures and backgrounds. Raw/final releases include UV, segmentation, 2D/3D grids and metadata. | MIT in author code repository. Background/document sources are listed separately. | Particularly practical route to measured folds and lighting while replacing page content with known ECG rasters. Code is an offline conversion reference, not a ready torch-only provider. |
| [DocSynth, ICDAR 2021](https://github.com/biswassanket/synth_doc_generation) | PyTorch layout-conditioned document synthesis trained on PubLayNet. | MIT in repository | Useful for layout diversity research. A learned page generator has no demonstrated guarantee of sample-exact ECG traces. Do not put it across the waveform or derive masks from its output. |
| [ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit), [paper](https://doi.org/10.1088/1361-6579/ad4954) | Real supplied waveforms rendered with ECG layout, text and paper damage augmentations. | BSD-3-Clause in author repository | Mandatory matched-waveform comparator. Match resolution, source records and degradation budget. Do not infer clinical superiority from feature counts. |
| [SyntheticDoc](https://github.com/tanguymagne/SyntheticDoc), [paper](https://arxiv.org/abs/2609.15503) | Physics-simulated meshes and path-traced document scenes. Author repository separates simulation, rendering and backward mapping. | MIT in code repository. Linked data/assets require their own recorded terms. | Strong current reference for offline generation. On inspection, rendered image, albedo, shading, UV, backward map and metadata were released, while normals and 3D coordinates were announced for later release. Do not promise unavailable annotation downloads. |

The name DocSynth is also used by unrelated software and by DocSynth-300K from
[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO). The latter concerns
synthetic document layout pretraining. Neither is a substitute for a calibrated
physical ECG image-formation model.

## Source-code observation that matters for integration

UVDoc's [`create_final.py`](https://github.com/tanguymagne/UVDoc-Dataset/blob/main/create_final.py)
uses a UV sampling field to place document texture onto a captured sheet and
combines it with paper appearance and background. It uses NumPy/SciPy/OpenCV as
well as torch and includes fixed blending/blur heuristics. Adopt the measured
geometry/illumination concept, not the claim that its compositing is a complete
physical camera. The current renderer already exposes an external geometry bank.
That is a useful extension point for an offline converter with documented UV,
mask, orientation and color-space conventions.

## Why the baseline can look artificial

These are engineering hypotheses from source inspection, not results of a blinded
human study. Several can coexist in the same image.

- Large, recognizably procedural surfaces and background patterns can dominate
  subtle paper and device cues. Repetition across a batch creates an easy shortcut.
- Strong shadows without a corresponding plausible occluder or coherent sheet
  geometry look painted onto a rectangle.
- Smooth gradients do not reproduce material roughness, printer variation,
  contamination, illumination and sensor noise at their different physical scales.
- Damage sampled independently of capture type creates improbable combinations.
  Native thermal print, laser copy, flatbed scan and phone photo are distinct
  acquisition processes. Clean examples should remain common where justified.
- Applying photometric operations to nonlinear display values changes energy and
  edge appearance. A generic gamma knob is not a complete color pipeline.
- Perfect textbook typography and trace styling on heavily aged paper can produce
  an inconsistent page even when geometry is accurate.

Do not address these problems by drawing a new physiological waveform. Original
signal samples, lead identity, timing and gain remain authoritative.

## Physical image-formation requirements

[W3C CSS Color 4](https://www.w3.org/TR/css-color/) defines nonlinear sRGB and its
linear-light counterpart. [PBRT's camera measurement chapter](https://pbr-book.org/4ed/Cameras_and_Film/Film_and_Imaging)
provides the radiometric image-formation reference. For a defined sRGB input
contract, decode before light integration and optical mixing, then encode for
display. Material albedo, scene radiance and final display RGB should not be
silently interchanged. The exact boundary must be documented even if the model
retains approximations.

[EMVA 1288 Linear 4.0](https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf)
Eq. 15 describes a linear sensor photon-transfer relationship between temporal
variance and mean signal, with dark and quantization contributions. A Gaussian
read-plus-shot approximation can implement that relationship, but does not
simulate Bayer sampling, demosaicing, denoising, tone mapping or a named phone.
Avoid claiming EMVA compliance without the required measurements.

Separate material-coordinate effects, light transport and camera effects:

1. Form print coverage in millimeters, including device-specific width and fading.
2. Apply paper/print condition in material coordinates, including local stains.
3. Project the paper using a consistent UV, visibility, depth and normal model.
4. Compute or import coherent illumination and contact/cast shadows.
5. Integrate optics and exposure in linear light, apply a sensor approximation,
   then the defined display and compression pipeline.
6. Derive lead masks from the reference geometry through the same projection.
   Distinguish geometric support, opaque occlusion and observation readability.

## Prioritized acceptance gates

All thresholds below are proposed engineering gates, not medical standards.
They describe evidence still needed, not tests already passed by this review.

| Priority | Gate | Concrete evidence |
| --- | --- | --- |
| P0 | Correct light integration | Independent black/white mixing reference: 50% linear radiance encodes to approximately 0.735357 sRGB. Verify constant-field preservation and normalized PSF impulse energy, separately from clipping. |
| P0 | Bound optical support | Compare every documented blur/motion range against a dense reference. Reject or explicitly document user overrides whose kernel energy is materially truncated. A fixed 9×9 kernel cannot be assumed accurate for arbitrary lengths. |
| P0 | Annotation geometry | Each visible lead instance has identity, window, polarity, mask and tight box in the final image. Repeated rhythm strips remain separate instances. Test complete occlusion, partial cropping and overlapping leads. Recompute boxes from final support or a certified geometric projection. |
| P1 | No periodic shortcut | Publish native-size crops of paper, table, grid and ink across at least 32 fixed seeds. Compare autocorrelation/spectral peaks with held-out real crops and inspect repeated assets. Human inspection remains required. |
| P1 | Measured asset bank | Offline manifest records source, license, checksum, units, UV orientation, color space and allowed uses. Separate measured shading from albedo and avoid applying an additional procedural light twice. Preload tensors on device before training. |
| P1 | Geometric consistency | Fiducial-grid projection and mask/image alignment agree within 0.5 output pixel in nonsingular visible interiors. Test visibility separately at boundaries and self-occlusion. For bent paper, publish metric strain and inverse-map failures. |
| P1 | Capture calibration | Capture a known printed page plus gray/color references across scanners and phones, exposures and paper conditions. Estimate correlated parameter distributions using a calibration partition and freeze them before evaluation. |
| P2 | Real ECG transfer | Train fixed YOLO and U-Net baselines with matched waveform counts and budgets. Compare clean synthesis, ECG-Image-Kit, current procedural renderer and measured-bank renderer on held-out real captures. Report per-lead detection AP and segmentation Dice/IoU, plus digitization SNR/voltage error where paired signals exist. |
| P2 | Realism assessment | Blinded expert review of full pages and native-size crops, stratified by capture class. Include failures and record-level confidence intervals. A real-versus-synthetic classifier is a shortcut diagnostic, not proof of indistinguishability or clinical utility. |

The [ECG-Image-Database publication](https://arxiv.org/abs/2409.16612) describes
35,595 images from software-generated pages with physical printing, damage and
capture, paired with source waveforms. It is an appropriate external-corpus
candidate. Preserve recording/patient groups across splits and never tune on the
held-out captures. Access, selected release and redistribution terms still need
to be confirmed before importing or bundling its files.

## Evidence boundaries and next deliverable

No external code or third-party assets were copied by this research task. No
real-image dataset was imported. Repository license labels were checked in the
author sources, but they do not automatically clear every separately downloaded
texture, font, mesh or dataset. Pin an exact revision and asset manifest when
adopting one.

The most valuable next realism deliverable is a small reproducible measured-paper
bank with correctly aligned ECG targets and an external test report. A compelling
gallery supports visual review, while held-out real capture transfer supports the
stronger claim that the renderer improves training.

## Optional offline Mitsuba route, added during integration

The optional route is implemented in `ecg_gpu/pathtrace.py`. Independent real
Mitsuba execution below confirms the reviewed paths can execute. This supplement
does not establish a universal pass for every configuration. The route stays
separate from the torch training hot path.

[Mitsuba's integrator reference](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_integrators.html)
describes `path` as a general path tracer with direct and indirect illumination.
Finite maximum depth is a documented approximation, so record integrator, depth,
sample count, seed and version. Its AOV interface includes UV and shape/primitive
IDs, but explicitly warns that integer IDs become fractional with partial
coverage or wide reconstruction filters. Averaged UVs can also mix unrelated
surfaces. Do not round such values into a lead mask.

For this application, derive labels from primary-ray paper intersections using
the same camera and first-visible-surface test as the rendered scene. Pixel-center
labels are valid geometric targets if documented as such. Soft coverage labels
require explicit subpixel sampling. Neither scheme is identical to a blurred or
Monte Carlo filtered observation at boundaries. Do not infer paper identity from
depth alone, since an occluder can be closer. Use the actual intersected shape.

[Mitsuba bitmap documentation](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html)
specifies that ordinary sRGB bitmap files are linearized by default. Tensor
`data` textures require `raw=True`, so provide already linear color values there.
Use raw data for UV, normal and roughness maps. Do not decode sRGB twice or mark
an encoded diffuse image as raw. Bilinear texture lookup provides interpolation,
not minification filtering. Fine ECG traces and grids need adequate rendering
sample counts and preferably a supersampling convergence check. Save radiance
before display encoding and explicitly record the final encoding/tone mapping.

[Mesh documentation](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html)
shows different vertical texture-flip defaults for OBJ and PLY. Set the convention
explicitly. [Camera documentation](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html)
uses right-handed sensors with camera +X pointing left, +Y up and +Z along the
view direction. Prefer the sensor's own ray generation over a separate assumed
computer-vision convention. Verify an asymmetric four-corner color/ID fixture in
RGB, UV and targets, including an oblique view.

[The BSDF reference](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html)
documents one-sided diffuse materials and the `twosided` adapter. A thin folded
sheet needs a defined back surface. If the ECG print is one-sided, using its
front texture on both sides creates a physically incorrect mirrored print on
the rear. Back appearance, transmission and bleed-through are distinct choices.

The official [Mitsuba LICENSE](https://github.com/mitsuba-renderer/mitsuba3/blob/master/LICENSE)
contains three BSD-style redistribution conditions and an additional paragraph
about voluntarily supplied enhancements. Preserve and reference the exact text,
rather than silently describing the dependency as MIT. No Mitsuba implementation
code was copied by this research task.

The proposed [Poly Haven Wood Table 001](https://polyhaven.com/a/wood_table_001)
is independently confirmed by its author page as CC0, photographed by Dimitrios
Savva and processed by Rico Cilliers, with a reported physical extent of 1.5 m.
Use the diffuse map as reflectance, not a lit beauty preview. Record the exact
downloaded map, resolution, hash and meter-to-UV scale. Normal, roughness and
displacement maps have different semantics and should not inherit sRGB decoding.
This is a real material asset, not evidence that the generated ECG capture is a
real photograph or that this table is representative of clinical environments.

Minimum offline-PBR acceptance: asymmetric UV alignment, correct front/back
ownership, opaque-occluder visibility, physically scaled grid, finite radiance,
texture filtering convergence, repeatable provenance, and a native-resolution
image/lead-mask/box overlay inspected together. Path tracing improves transport
coherence but does not turn a procedural mesh into a validated paper mechanics
solver or establish photographic indistinguishability.

### Independent implementation checks

Actual CPU execution used a constructed page, independent quadrant targets and
Mitsuba's real renderer, not a mocked path tracer. A folded 96×96 capture at
8 image samples/pixel and 2×2 label samples/pixel correctly matched red, green,
blue and white source quadrants to their corresponding masks. Mean RGB inside
the four masks above 0.95 coverage was respectively `(0.758,0.002,0.002)`,
`(0.001,0.520,0.001)`, `(0.008,0.007,0.822)` and `(0.550,0.549,0.548)`.
This independently supports UV orientation and image/label agreement in visible
interiors. The physical width `canonical_width / pixels_per_mm` is consistent
with bitmap pixel-edge coordinates and `align_corners=False` sampling.

Review identified issues and sent them to the coordinator before integration:

- The initial trapezoidal unit-tangent mesh produced up to 2.991% discrete
  edge-length contraction for the maximum 1.2-radian fold, versus below 0.005%
  for the curl. Segment-midpoint unit tangents or a documented refinement bound
  are needed before claiming discrete arc-length preservation.
- The initial pen material used an untyped RGB vector rejected by Mitsuba.
  `occluder=True` raised a real scene-construction exception. A typed RGB spectrum
  fixes the defect. With that correction applied only in a temporary in-memory
  test wrapper, pen visibility removed 24, 2 and 10 paper pixels in flat, fold
  and curl captures at 64×64. Occlusion extent depends on the paper geometry.
- Replay metadata initially omitted the full camera/light configuration and
  canonical content hash. Persist these when claiming a reproducible capture.

All three findings were corrected and independently rechecked after integration.
For 271×203.25 mm sheets at the maximum 1.2-radian bend, maximum relative
horizontal edge-length error was 3.219e-6 for flat, 2.503e-6 for curl and 2.146e-6
for fold. The midpoint-tangent mesh therefore passed the proposed 1e-4 discrete
edge-strain gate across these cases, within float32 coordinate precision.

Actual Mitsuba renders at 64×64, one image sample/pixel and 2×2 label rays/pixel
completed with and without the pen for all three surfaces. The corrected pen
removed 19.25, 20.00 and 21.25 equivalent paper pixels for flat, fold and curl,
respectively. These fractional counts are stratified pixel coverage differences,
not annotation IDs inferred from filtered RGB. This closes the earlier material
exception and below-paper placement findings for the checked configurations.

Independent assertions confirmed exact `asdict(config)` persistence, the expected
source-reference hash, the current implementation-file hash, a 64-character
canonical-texture hash and retained sample rate for all six rendered cases.
This closes the earlier provenance omissions. Cross-version/device bitwise
replay was not established.

The 128-sample/pixel three-case gallery at
`build/round2-pathtraced/pathtraced-gallery.jpg` was visually inspected. Flat,
curled and folded pages, table grain, pen shadow and mask/box overlays are
coherent in the displayed images. Stronger realism limits remain: the page shape
is prescribed, the paper material is approximate and the sample gallery is not
a blinded real-image comparison. Native resolution, photometric calibration,
severe self-occlusion and held-out clinical-device captures remain separate
evaluation work.
