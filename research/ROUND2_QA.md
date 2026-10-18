# Independent adversarial quality review, round 2

Review date: 2026-10-18. Reviewed the repository implementation, existing tests,
original `research/assets/scenarios.jpg`, the intermediate
`build/round2-physics-after/scenarios.jpg`, and new training/export adapters.

## Verdict and evidence boundaries

The repository already supplies usable per-lead and per-occurrence supervision.
This is a concrete engineering capability. It does not demonstrate that its
images are indistinguishable from photographs or that a model trained on them
will generalize to hospitals. No claim of 100% realism or superiority over all
other ECG generators is supported by the evidence reviewed here.

Three different gates must remain separate:

1. Numerical correctness: geometry, signal identity, mask ownership and exports.
2. Human visual quality: how the rendered paper, print and capture look.
3. External validity: blinded real-versus-synthetic assessment and downstream
   performance on independent real photographs and scans.

Passing the first gate does not imply passing either of the others.

## Baseline results

Ran once before integrating round-2 implementation changes:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
Ran 102 tests in 25.933s
OK (skipped=3)
```

The three skipped checks require CUDA. CPU compilation and CPU training checks
cannot establish GPU correctness or speed. The integrator owns the final
post-integration suite result because other agents are editing shared files.

## What looks artificial in the original gallery

The following are direct qualitative observations of the checked-in original
gallery, not scores from a real-image benchmark:

| Priority | Visible cue | Implication |
| --- | --- | --- |
| High | Broad pink/green casts dominate several photographs | Reads as a color overlay rather than a neutral paper document under a plausible camera pipeline |
| High | Backgrounds are featureless pastel planes or regular wood bands | Repeated mathematical texture is an easy synthetic domain cue |
| High | Folded case resembles a warped image with a broad smooth tint | A fold needs consistent sheet geometry, local highlight, shadow and contact behavior |
| High | Header is sparse, generic and repeated | Real ECG devices have distinctive metadata typography, spacing, separators and print conventions |
| Medium | Traces have highly uniform thickness and darkness | Printer transfer, paper coating and sampling vary spatially and by device |
| Medium | Coffee mark is a conspicuously regular ring | Real liquid residue has broken edges, irregular wetting and heterogeneous pigment |
| Medium | Demo beats and leads repeat stereotyped patterns | These examples assess rendering, not physiological diversity or diagnostic realism |

Adding stronger blur, noise, damage or texture is not by itself an improvement.
It can obscure synthetic cues while making the acquisition less physically
plausible and the waveform less useful.

## Physical issue found during source review

In the baseline renderer, opaque cover, pen scribble and artifact text are
composited after paper projection in camera/image coordinates. That is suitable
for an object between camera and paper, but a handwritten mark or printed stamp
on a folded sheet should follow the paper coordinate field. The reviewed
implementation needed an explicit distinction between document-attached marks
and camera-plane occluders. This finding was sent to the integrator and physics
owner. The round-2 fix moves pen and stamp ink into material coordinates before
projection while retaining the opaque rectangle as a camera-plane occluder.

Independent review caught a regression in the first version of that fix:
zero-padding the material-visibility mask gated fractional paper silhouette
coverage twice. A specific test failed on 291 pixels, with maximum error 0.25.
The owner changed material-visibility sampling to border padding. The same test
then passed. This demonstrates a corrected numerical contract, not realism.

The surface and optical tests do verify useful limited claims, including unit
normals, bounded metric error, inverse projection residual, shutter integration
and read-plus-shot noise variance. These are independent analytical checks.
They do not validate thermal-paper reflectance, material thickness, calibrated
camera response or the statistical distribution of actual ECG captures.

## Supervision and training review

The task needs two related targets:

* Twelve canonical lead channels, preserving overlaps as independent sigmoid
  channels. A categorical softmax mask loses ownership at crossings.
* Separate printed occurrences for detection. A repeated lead-II rhythm strip
  is another object with the same class, not a box spanning both disconnected
  strips. Cabrera inversion changes the displayed sign, not canonical identity.

Trace-tight boxes enclose waveform pixels. They are not document-cell boxes
and do not automatically include lead labels, calibration pulses or metadata.
Consumers must use a consistent definition when annotating real validation
images. Visible targets describe observed support, while projected/amodal
targets require a different training objective. Strong blur does not change the
geometric label and can make an otherwise valid trace visually unrecoverable.

The legacy renderer metadata uses zero boxes for empty occurrences. The
supervision interface uses `-1` and an explicit validity flag. Export adapters
must use the validity flag, never infer existence from a coordinate alone.

The default `valid` mask excludes background outside the paper and occluded
pixels. Multiplying all segmentation loss by it ignores tabletop negatives.
That is a deliberate paper-only objective, not a universal segmentation loss.
For full-frame segmentation, the caller must choose supervision support that
also teaches outside-paper background when appropriate.

The YOLO export is detection, not YOLO segmentation. A disconnected thin ECG
trace must not be silently converted into a filled polygon. U-Net uses dense
independent masks directly. The torchvision adapter must reserve class zero
for background when the selected detector does so. This issue was raised with
the adapter author, who agreed to a configurable default offset of one for
torchvision while preserving zero-based YOLO classes.

## Independent round-2 regression checks

Added `tests/test_round2_quality.py`, with six tests at review time:

* Brute-force coordinate enumeration for tight boxes in phone, fold and damaged
  captures, independent of the production bounding-box helper.
* Explicit canonical identities and sign of Cabrera aVR, plus repeated rhythm
  occurrences and canonical unions on unwarped images.
* Hand-derived normalized XYWH values in a non-square image, border clipping,
  repeated class instances and invalid-class filtering.
* Actual offline PNG/label export, annotation numeric round-trip, file SHA256
  verification, patient grouping and stable splits under input reordering.
* Rejection of identical source hashes assigned to different patient groups,
  before the exporter creates a directory.
* Preservation of fractional silhouette coverage when no document marks are
  present, rather than incorrectly squaring that coverage after projection.

All six passed in 0.829s on CPU after the coverage fix. The supervision owner separately provides a
broader all-layout mask and occurrence audit.

Group splitting can prevent known source/patient overlap only when the caller
supplies truthful stable group IDs. It cannot discover hidden patient identity
or unrelated recordings of the same patient. All augmentations and windows of
a patient must stay in that patient's split before generation.

## Intermediate visual re-review

In `build/round2-physics-after/scenarios.jpg`, periodic wood stripes are reduced
and the stamp follows the sheet. Those are useful improvements. The sparse
generic header, repeated artificial waveforms, very smooth tabletops, regular
coffee annulus and broad folded-page pink cast remain strong synthetic cues.
The folded-page case still needs clearer localized crease and contact cues.
This was an intermediate gallery before final scenario parameter integration,
so the integrator must inspect the final gallery as well.

## Independent optional path-tracer review

The coordinator subsequently added `ecg_gpu/pathtrace.py` as an optional offline
Mitsuba backend. It is separate from the on-device renderer and its CPU/file
operations must not enter a training forward pass. Added six independent tests
in `tests/test_pathtrace_quality.py`, which passed with Mitsuba available:

* Asymmetric red, green and blue canonical fiducials are rendered and compared
  directly with interior mask pixels. This detects a vertical UV flip or
  mismatch between RGB and supervision. Boxes use a brute coordinate oracle.
* Actual sensor rays are independently intersected with analytical planes to
  check paper and foreground-occluder visibility. The oracle does not use the
  renderer's scene-intersection answer.
* A camera viewing a two-sided paper rectangle from behind receives no front
  print labels, even though it intersects the sheet.
* Meter-scale sheet dimensions, explicit top-left UV corners, face orientation
  and integrated mesh length are checked. The earlier trapezoidal integrator
  contracted a 300 mm folded sheet by 0.167 mm at a 1 radian bend. The coordinator
  replaced it with midpoint unit-tangent edges. The final tests bound relative
  strain of every triangle edge below `1e-4` across flat, curl and fold surfaces
  at the maximum 1.2 radian bend, including diagonal mesh edges.
* The public `pathtrace_capture(..., occluder=True)` branch really renders a
  folded sheet with its cylinder material. This catches invalid Mitsuba BSDF
  property construction that a custom analytical test scene cannot exercise.
  Compared with the same configuration without the pen, paper support decreases
  and mask coverage cannot increase.
* Full camera/light/surface configuration and source shape/dtype/hash are
  asserted in returned provenance, along with implementation and texture hashes.

All six tests passed in 1.172s on CPU. The four tests requiring Mitsuba skip
when the optional dependency is absent.
Passing these checks establishes useful primary-ray alignment and physical
units. It does not establish realistic thermal-paper BSDFs, folds drawn from
real document distributions, camera/lens calibration or realistic scene assets.
The default scene remains a simplified prescribed sheet on a plain table, with
an optional real texture asset rather than a measured full scene.

## Final path-traced gallery visual review

Inspected `build/round2-pathtraced/pathtraced-gallery.jpg`, including the flat,
curl and fold captures and their mask/instance-box overlays. Compared with the
original gallery, the real wood texture, directional shading and consistent
three-dimensional sheet silhouette provide a visibly more convincing scene.
The fold changes printed grid and traces coherently, and the pen causes actual
occlusion instead of merely painting an overlay. Those are specific advances.

The result still has conspicuous synthetic cues. The curl is a mathematically
uniform sheet, the fold resembles a clean tent shape, paper edges lack thickness
and wear, and the blue cylinder resembles a simple rod rather than a detailed
pen. The document still has the same sparse generic header and artificial demo
waveforms. The page is grey and somewhat grainy in this gallery, so calibrated
exposure, material reflectance and sufficient render sampling remain relevant.
One wood tabletop reused across three shapes does not establish environment
diversity. There is no basis here to claim indistinguishability from real
photographs or performance superiority over alternative generators.

## External validation still required

Build an independent reference collection with device/vendor, print format,
paper type, capture mode and adversity annotations. Split by patient and by
device/site where possible. Do not tune parameters on the final test collection.

Measure grid geometry, line-spread/edge response, paper and grid color,
background spatial spectra, noise-versus-intensity, fold deformation and shadow
profiles against the corresponding strata of real images. A single pooled
image-distance score is insufficient.

Use blinded assessment on matched real and synthetic ECG crops and full pages,
with multiple independent reviewers and uncertainty intervals. Test whether a
domain classifier generalizes across unseen devices. Run controlled downstream
ablations for real-only, synthetic-only and mixed training, reporting detector
performance, per-lead segmentation, waveform recovery error and calibration.
Report all failure strata. Without this evidence, describe the output as
procedural physically motivated ECG rendering, with measured engineering
contracts and explicitly unverified photographic equivalence.
