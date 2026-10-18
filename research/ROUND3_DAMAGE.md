# Round 3: damage to physical paper support

The damage revision replaces the old single symmetric notch, decorated with an
absolute sine, with a material-space damage model. It adds ragged tears on up
to four edges, shallow edge abrasion, three irregular internal punctures, three
finite open cuts and a removed corner. These remove substrate rather than
painting a white or dark artifact over the ECG.

## Interface and units

`DocumentScene.paper_mask(field, parameters)` returns binary surviving material
with shape `[B,1,H,W]`. `field` maps sensor pixels into the original paper UV.
The mask uses `x=UV.x` and `y=UV.y*height/width`, so a physical radius has the
same size along either axis. Size `.01` means 1.5 mm on a 300 mm wide sheet.
Centers remain normalized sheet UV. Different ECG durations or print speeds
can change the physical sheet width, so the renderer calibration and provenance
must remain authoritative when converting these normalized sizes to mm.

| Parameter | Shape | Default | Meaning |
| --- | --- | --- | --- |
| `tear_active` | B | Existing sampling | Enable tears |
| `tear_side` | B | Existing sampling | First edge, right/left/bottom/top |
| `tear_position` | B | Existing sampling | First tear center along edge in normalized UV |
| `tear_size` | B | Existing sampling | Maximum nominal depth in half-page-width units |
| `tear_count` | B integer | 1 | Number of edge tears, 0 to 4 |
| `tear_raggedness` | B | .65 | Shared damage roughness, 0 to 1 |
| `edge_wear_strength` | B | 0 | Distributed shallow erosion of outer edges |
| `edge_wear_depth` | B | .008 | Maximum edge erosion depth |
| `hole_active` | B,3 | 0 | Three independently enabled irregular punctures |
| `hole_center` | B,3,2 | Sample-derived | Puncture centers in original sheet UV |
| `hole_radius` | B,3,2 | .025 to .055 | Ellipse half-axes, physical half-width units |
| `hole_angle` | B,3 | Sample-derived | Ellipse rotation in radians |
| `cut_active` | B,3 | 0 | Three independently enabled open kerfs |
| `cut_center` | B,3,2 | Sample-derived | Cut centers in original sheet UV |
| `cut_length` | B,3 | .12 to .30 | Straight segment length before end caps |
| `cut_width` | B,3 | .003 to .006 | Actual missing strip width |
| `cut_angle` | B,3 | Sample-derived | Cut rotation in radians |
| `missing_corner_active` | B | 0 | Remove a corner of the sheet |
| `missing_corner_index` | B integer | Sample-derived | 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left |
| `missing_corner_size` | B,2 | .12 to .222 | Nominal missing corner extents, physical half-width units |

The new parameter keys are always sampled, but the additional damage types are
opt-in. Their seeds derive from existing random surface samples. No additional
random draw is consumed. With every damage activation zero, support is exactly
one, including points outside the source frame. Source-frame pixel coverage is
handled separately by the renderer and must not be applied twice.

`tear_raggedness=0` gives analytical ellipses, straight capsule cuts and straight
corner cuts. This covers manufactured punctures and low-roughness cut edges
without pretending that the same process describes torn fibres. Positive
roughness varies contours over several material scales. The edge profile sums
four incommensurate frequencies, with bounded amplitudes, rather than repeating
a single regular sawtooth. Nearby tears reuse material fibre variation to keep
spatial coherence. Irregular punctures use three angular contour scales.

A static `DocumentScene(..., damage_detail='rich')` evaluates all types. The
explicit `damage_detail='edge'` alternative evaluates just the first edge tear
and ignores the advanced damage parameters. It avoids per-batch host/device
synchronization and supports fast training configurations where those types
are intentionally absent. Both variants remain entirely torch and on device.

Recommended renderer validation bounds are alpha values in [0,1], roughness
in [0,1], tear count 0 to 4, corner index 0 to 3, centers in [-1,1], positive
hole radii at most .25, positive cut length at most 1, positive cut width at
most .05, positive corner extents at most .6 and edge-wear depth in [0,.05].
These are per-component bounds. There is no claimed universal area-removal
bound for arbitrary combinations or extreme page aspect ratios. Training
scenario sampling should choose realistic combinations rather than activating
all maximum settings simultaneously.

## Alignment and shadows

Missing paper is expressed in original material coordinates. Perspective,
rotation and non-flat surfaces therefore move holes and cuts with the document.
The same binary paper mask is available to the RGB compositor, trace and
instance validity, and tabletop shadow caster. A hole does not cast a paper
shadow. A stain is a different appearance process and does not remove paper.
The existing conditional frame-coverage correction remains necessary so that
visible trace coverage at the source-frame silhouette is not squared.

The accompanying tests directly check the shadow projector using a known
vertical light, where projection must exactly equal the damaged caster. They
also independently check analytic ellipses, capsule distance, corner orientation,
material metric units, material inverse-map movement and simultaneous tears.
The integration reviewer separately owns image/lead/box alignment tests.

## Verification

`python -m unittest tests.test_damage_r3 -q` passed 11 tests. Combined with
`tests.test_physical_scene`, all 21 tests passed. The damage kernel also traces
as a single full torch graph. A separate real Inductor compile of the rich
kernel at 768 by 576 produced bitwise-identical masks to eager execution.
The eager backend graph test is included in the regular suite to avoid adding
an expensive compiler build to every CPU CI job.

CPU-only timing, torch 2.14.1+cpu, two torch threads, batch one, seven warm
evaluations, excluding sampling and other renderer work:

| Image | Edge detail | Rich detail |
| --- | ---: | ---: |
| 144 by 96 | .870 ms | 6.491 ms |
| 768 by 576 | 14.501 ms | 97.687 ms |
| 1500 by 1000 | 51.919 ms | 314.089 ms |

The rich eager path is more costly even with activation tensors zero, because
it avoids synchronized host decisions. The explicit static edge path reduces
that cost. A separate full Inductor experiment on an earlier rich revision took
7.38 seconds to compile and 159.7 ms warm at 768 by 576. This was slower than
eager on this CPU and is not advertised as a performance improvement. CUDA
measurements were unavailable. These figures do not measure total rendering
or training throughput.

## Limits and evidence

This is a bounded geometric support model, not a solver for fracture mechanics,
fibre pull-out, burn propagation or physically separated sheet fragments. A
finite open cut models an actual kerf, not a zero-width incision with intact
paper on both sides. The 2.5D document solver cannot create rolled-over torn
flaps or the interior thickness of a ripped edge. Texture and microscopic fibre
appearance need separate calibrated assets. Procedural roughness does not
establish photographic indistinguishability.

The research comparison identified [DocCreator](https://github.com/DocCreator/DocCreator)
as an important reference because it includes masks sampled from real document
defects. A future empirical defect-mask bank would improve morphological
coverage beyond procedural ellipses and edge tears. Actual topology-changing
thin-sheet dynamics are a separate research problem, illustrated by
[Pfaff et al., Adaptive Tearing and Cracking of Thin Sheets](https://tobiaspfaff.com/publication/pfaff-2014-adaptive/).
Neither result is claimed to have been reproduced by this revision.
