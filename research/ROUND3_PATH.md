# Round 3: irregular and physically missing paper in the offline 3D backend

## Implementation

`ecg_gpu/pathtrace.py` now exposes three additional surface families through
`PathTraceConfig.surface`. `multifold` uses seeded, unequally spaced rounded
creases with different signed angles and widths. Like the earlier flat, curl
and fold modes, it integrates unit tangents and preserves the material metric.
The creases in this ruled family span the sheet. They are not a cloth solver.

`wrinkled` and `crumpled` instead combine finite irregular crease ridges,
rounded tips, flatter adjacent facets and correlated submillimeter waviness.
Lengths and amplitudes are expressed in meters. A symmetric static relaxation
projects edges and diagonals toward their original material lengths. These
modes retain measurable residual strain. Each render reports the maximum
absolute edge strain rather than claiming exact inextensibility. They do not
simulate elastoplastic shell mechanics, crushing history or collisions.

The new configuration fields are `geometry_seed`, `surface_strength`,
`fold_count`, `wrinkle_amplitude_m`, `missing_corner`, `edge_tear`,
`hole_radius` and `camera_up`. Camera roll comes from a real camera up vector
used by the same perspective sensor that supplies the ground-truth rays.
`key_origin`, `key_target` and `key_size` specify an actual rectangular area
light in meters. `key_size` is full width and height, with backward-compatible
defaults of 0.36 by 0.22 meters. Raking illumination reveals the same geometric
relief that casts shadows, without a separate painted crease-shading layer.

Damage clips actual triangles from the sheet. Missing corners, irregular
edge notches and irregular interior holes are shared by light transport,
shadows, visibility, trace masks and bounding boxes. Optional canonical
`targets['paper_support']` of shape `[1,1,H,W]` is sampled at material vertices.
The bilinearly sampled 0.5 isoline clips each intersected triangle. This support
controls topology only. It does not multiply fractional target coverage again.
The separate canonical `valid` factor is applied once to sampled labels.

An independent visual review rejected the first implementation because removing
whole coarse triangles produced conspicuous staircase edges. Production capture
now refines the original piecewise-linear physical mesh and clips triangles on
interpolated support boundaries. Shared edge intersections preserve XYZ and UV
continuity. The physical solve stays at 129 by 65, while damaged-capture
topology adapts to output resolution. A 768 by 576 capture uses a 385 by 321 grid
before clipping. A 1024 by 768 capture uses 513 by 385. Refinement is capped at
769 by 385, with finite segment resolution explicitly recorded. Direct
`paper_mesh` defaults remain compatible with earlier metric tests.

Backside intersections remain excluded from printed-trace supervision.
The rear surface has separate unprinted reflectance. A rigid pen is raised
against the sampled local sheet profile rather than positioned below a fold.

## Provenance and numerical evidence

Metadata includes the full configuration, geometry seed, actual mesh hash,
vertex and triangle counts, maximum measured edge strain, canonical texture,
reference, target and identity hashes, optional paper-support hash, background
hash, implementation hash and software versions. Damage boundaries explicitly
state their interpolation and mesh-resolution limit. The refined 768 by 576
hero contains 124,975 vertices and 235,550 retained triangles after clipping.
An independent camera projection of its 1,876 boundary segments measured a
maximum segment length of 1.99015 image pixels and a 95th percentile of 1.65777
pixels. This is evidence for that configuration, not a universal guarantee for
all camera poses. The 32-spp prototype rendered in 6.05 seconds on the available
CPU environment. It does not resolve individual paper fibers or torn-edge
thickness.

Fourteen focused tests cover CPU Mitsuba 3.9.1, including the six existing
independent checks. The eight new tests cover seeded geometry, positive
front-facing facets, measured metric error, invalid configuration, actual
removed topology, real rendered paper disappearance, source-support cuts,
retained 12 lead masks and 15 instance masks, and repeated geometry/radiance.
The follow-up adds an independent analytical clipped-rectangle area oracle,
an exact material-boundary position check, sub-nanometer refinement continuity,
and an actual refined-damage render with a pen and configurable raking light.
RGB replay permits 2e-7 absolute floating-point reduction differences.
Geometry and label sampling remain deterministic for a fixed scene and
software environment.

The full-resolution adverse gallery also exposed a genuine numerical defect:
float32 interpolation could collapse a boundary UV edge while its XYZ edge
still retained a nonzero floating-point ULP. Dividing by that zero material
length produced infinite metric metadata, correctly rejected by strict JSON
export. Interpolation now uses float64, then validates both UV and XYZ after
float32 export quantization. Unresolvable sliver triangles are discarded using
an explicit 64-times-float32-epsilon edge threshold, approximately 2 micrometers
on this page. Metadata records the number discarded and both resolution
thresholds. Physical strain is computed in float64 across every retained edge,
with explicit errors for any remaining zero or nonfinite metric. No infinite
strain is silently replaced or omitted. Regression tests include a constructed
collapsed-UV triangle and the actual oblique-fragment source-support parameters
rendered from a 1024 by 768 canonical page, with RuntimeWarnings treated as
errors and `json.dumps(..., allow_nan=False)`. All eight new tests passed after
this fix in 15.625 seconds.

A separate extreme-parameter sweep used 12 folds, bend 1.2, strength 1.5 and
seeds 1, 2, 3, 41 and 1973 on a 300 by 200 mm page. Maximum absolute triangle
edge strain was 0.00000533 for multifold, 0.01058 for wrinkled and 0.02047 for
crumpled. These are explicit limitations of the prescribed relaxed geometry,
not evidence for real paper mechanics. The normal/default strength is milder.

## Example capture configuration

```python
PathTraceConfig(
    surface='crumpled', bend_radians=1.1, surface_strength=1.2,
    fold_count=7, geometry_seed=41, wrinkle_amplitude_m=.00035,
    missing_corner=.22, edge_tear=.14, hole_radius=.07,
    camera_origin=(.035, -.16, .39), camera_up=(.12, 1., 0.),
    key_origin=(-.24, -.09, .12), key_target=(0., 0., 0.),
    key_size=(.16, .09), key_radiance=12., environment_radiance=.18,
    width=768, height=576, samples_per_pixel=128,
)
```

## Outstanding realism limits

The geometry is an augmentation model and has no external real-image or
human-blind realism benchmark. Crease profiles and waviness are prescribed.
There is no self-contact, paper thickness, fracture dynamics or measured
elastoplastic material law. Strong crumpling retains percent-level strain in the measured sweep.
Interpolated triangle clipping gives finite linear boundary segments, and the
refinement cap limits quality at extreme output sizes and camera distances.
Fine torn fibers, translucent damaged edges and measured captured-surface
banks remain future work. Existing shape laws and backward-compatible calibration tests were
preserved without regenerating fixtures.
