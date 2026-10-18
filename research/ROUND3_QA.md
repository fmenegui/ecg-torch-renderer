# Independent adversity review, round 3

Review date: 2026-10-18. Scope: folds, crumpling, missing paper, cuts, aging,
liquid deposits, projective capture and preservation of training supervision.
Read `AGENTS.md` and the round-2 quality/supervision reports before review.
The quality reviewer edits only this report and `tests/test_adversity_r3.py`.

## Evidence boundaries

Algorithmic correctness, visual plausibility and measured real-image
generalization remain separate claims. These checks establish selected
material/geometry contracts. They do not establish photographic equivalence,
chemical aging accuracy, calibrated paper mechanics or downstream superiority.
No blinded real-versus-synthetic study was available for this revision.

## Independent acceptance checks

The new tests use direct coordinate enumeration and analytical geometry rather
than calling production bounding-box helpers to construct expected answers.

| Check | Independent reference | Result at initial integration |
| --- | --- | --- |
| Perspective differs from crop | Three hand-evaluated projective point correspondences, variable local scale, constant crop scale | Passed |
| Invalid camera horizon | Projective denominator crossing zero rejected before division | Passed |
| Interior missing paper | Exact ellipse in physical page-width units, including page aspect ratio | Passed |
| Finite cut | Capsule signed-distance formula, separate from damage implementation | Passed |
| Missing corner | Corner removed while sheet center/opposite corner survive | Passed |
| Irregular rotated damage | Material hole centers located through returned inverse field after rotation and perspective | Passed |
| Per-lead and occurrence boxes | Brute-force enumeration of every occupied pixel, exclusive right/bottom edges | Passed |
| Lead/occurrence preservation | 12 lead channels, 15 printed slots, removed material owns no trace target | Passed |
| Material off | Every substrate, deposit and print-retention factor equals one exactly | Passed |
| Aging color direction | Blue reflectance decreases more than green/red, print retention remains unchanged | Passed |
| Liquid variation/replay | Unequal equal-radius deposits, finite bounded factors, seeded field replay | Passed |
| Integrated material/GT separation | Aging, coffee, water, foxing and fading change RGB while all masks and visibility stay bit-exact | Passed |
| Source and metadata | Exact original signal preserved, no dense image fields added to metadata | Passed |
| Checkpoint replay | Restored random-call counter reproduces next complete training batch | Passed |
| Multiple-fold metric | Triangle edge lengths independently reconstructed from UV and physical dimensions | Passed |
| Missing mesh topology | Surviving triangles subset original topology, guaranteed interior hole lacks triangles | Passed |
| Actual 3D capture | Real Mitsuba render of damaged crumpled sheet, all lead/instance boxes independently checked | Passed |
| Fast-path capability contract | Edge-only mode rejects five kinds of advanced damage instead of silently ignoring them | Passed |
| Zero-sized padding | Zero dimensions accepted for inactive slots, rejected when those slots become active | Passed |
| Physical-removal activation | Fractional tear, hole, cut and corner flags rejected instead of becoming translucent geometry | Passed |

Execution after material integration, final mesh relaxation and parameter-bound
validation:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_adversity_r3.py -v
Ran 11 tests in 6.464s
OK
```

The integrator subsequently tightened the public damage validation. Three
additional targeted checks passed in 1.041 seconds with two OMP/MKL threads,
without concurrently invoking Mitsuba while the full integration suite ran.
The test file now contains 14 checks. This separate execution is not reported
as a complete 14-test run. After final implementation changes, the integrator
ran the full suite, including these checks, and reported **194 tests run,
190 passed, four CUDA checks skipped**, in 48.843 seconds. No new tests were
run concurrently with that integration pass.

The actual path-tracing test ran with Mitsuba installed on CPU. It skips when
the optional dependency is unavailable. CUDA execution remains unverified in
this environment. The final full-suite result above is the integrator's
record, distinguished from this reviewer's targeted executions.

## Numerical and modeling observations

1. A fixed-seed Mitsuba replay produced a maximum RGB difference of
   `1.1920928955078125e-7` on the first checked rerun. This is FP32 radiance
   accumulation variation. The test allows `3e-7` absolute RGB error while
   requiring bit-exact mesh hash, target masks and target boxes. Claims of
   universal bit-exact path-traced image replay would be too strong.
2. The new multifold sheet preserves the tested material triangle edge lengths
   to relative error below `1e-4`. This bound applies to the ruled multifold
   construction, not to arbitrary crumpled geometry. Crumpling uses prescribed
   relief and static distance relaxation, with residual strain recorded by
   the backend. It is not calibrated cloth or thin-shell mechanics.
3. Geometric cuts remove substrate. Transparent stains, thermal fading and
   yellowing instead change appearance and preserve the geometric trace target.
   A faded but geometrically valid trace is not a promise of readable signal.
4. Mask thresholding and boxes must use final visible masks. Transforming only
   box corners would fail for nonlinear wrinkles, missing corners and internal
   holes. The exercised implementation derives boxes after final visibility.
5. The topological damage boundary is discretized by the 3D mesh resolution.
   A physical opening is better than a painted black shape, but coarse triangle
   edges can still create visual cues and quantize small slits.

## Visual review

Inspected `build/material-r3-review.jpg`, the complete 15-case gallery at
`build/round3-scenarios/scenarios.jpg`, and native 1280-by-960 images for
`crumpled_photo`, `folded_archive` and `damaged_archive` at this checkpoint.

Specific improvements are visible. Tears have unequal boundaries. Internal
holes expose the same tabletop as external missing regions. The incomplete
bottom-right corner differs visibly from a rectangular camera crop. Oblique
capture produces a clear keystone shape rather than just rotation. The coffee
contact line varies in radius and pigment instead of forming the old exact
annulus. Aging has unequal edge penetration and broad variation.

Several visual limitations remain. In the real-time gallery, `crumpled_photo`
still reads mainly as a lightly warped almost-flat sheet. `folded_archive`
has broad bowing with weak persistent local crease cues. This observation was
sent to the integrator before final selection. A real 3D crumpled capture or
controlled raking-light example is a better demonstration of ridge structure.
The review does not recommend hiding the problem under stronger noise or tint.

At gallery overview size, narrow cuts can resemble a dark drawn line. Their
actual geometric opening is verified numerically, but missing fiber thickness,
cut-lip relief and local edge scattering still reduce physical plausibility.
The component foxing example has soft, broad yellow spots. Realistic spot-size
distributions and paper fibers still need measured references. The same wood
table repeated throughout this gallery, sparse generic `ECG RECORD` header,
and uniform printing remain recognizable synthetic cues independent of damage.

Also inspected initial real-3D renders `stored_folds.png`,
`crumpled_torn.png`, `wrinkled_coffee.png` and the `crumpled_torn-targets.png`
overlay in `build/round3-pathtraced/`. The stored folds have coherent shape,
shading and print perspective and are visually clearer than the real-time
fold example. The damaged/crumpled and coffee sheets still have subtle relief.
The colored supervision overlay follows printed ink and stops at missing paper.

One concrete high-priority visual defect was reported immediately: in the
initial 129-by-65-mesh damaged capture, hole and tear boundaries form obvious
triangle/grid stair steps, around 8 to 15 output pixels per segment. This is
especially conspicuous against a smooth otherwise intact sheet. It does not
break primary-ray target alignment, but it reduces realism in exactly the
missing-region cases requested. The reviewer recommended refining damaged
mesh boundaries before using this capture as a main visual example.
The integrator and 3D owner were notified before final release selection.

The 3D owner subsequently replaced whole-face deletion in the capture backend
with adaptive grid refinement and shared-edge support-boundary interpolation.
Inspected the 768-by-576, 32-sample prototype at
`build/round3-pathtraced-refined/crumpled_torn-refined.png`. The large stair
steps disappeared visibly. The internal hole and external tears now have
continuous irregular silhouettes. The raking key light reveals local ridges
that were almost invisible under the previous broad ambient light. These are
direct, specific corrections to the review findings. The prototype remains
grainy at 32 samples and needs final higher-sample inspection.

## Final release inspection

Inspected `build/round3-pathtraced-release/pathtraced-gallery.jpg` after all
four 1024-by-768, 192-sample captures and their target overlays completed.
The final publication artifacts are the release directory, rather than the
earlier intermediate/final-named render attempts described above.

The large damaged-boundary stair steps are resolved. Stored folds visibly
change the page silhouette, print direction and shading together. The crumpled
page has visible localized relief under the raking key light. Oblique fragments
combine keystone perspective, rotation, edge wear and missing substrate
without separating the print from the page. Colored masks and printed-instance
boxes follow visible waveform support and terminate at the surviving paper.
These are specific improvements over the round-2 and early round-3 examples.

Remaining cues include a somewhat flower-like harmonic internal hole, broadly
triangular large tears, absent edge fibers and thickness, sparse device-agnostic
headers, uniform printing and repeated use of one tabletop. The crumpled sheet
still represents a conservatively deformed opened document rather than an
arbitrary deeply crushed sheet. The result is suitable for release as an
improved procedural generator, with these limits stated explicitly. It does
not warrant a claim of photographic indistinguishability or universal realism.

Loaded the complete release JSON with a parser that rejects NaN/Inf. Compared
all four recorded implementation hashes to the actual `ecg_gpu/pathtrace.py`
SHA256, `aa543f1f8ba06bcd75f2fd8fbadb584775a651ec3b70d397c2d35cb65ea30b5b`.
Every hash matched. The recorded maximum absolute edge strains are finite:

| Case | Maximum absolute edge strain |
| --- | ---: |
| Stored folds | 0.00000332644 |
| Crumpled and torn | 0.00833316720 |
| Wrinkled coffee | 0.00328171026 |
| Oblique fragments | 0.00232245939 |

These are geometric distortion measurements for the actual generated meshes,
not statistical estimates of real paper mechanics. The integrator caught an
earlier zero-UV-edge clipping artifact through strict JSON export validation
before publishing a dataset. The backend now handles interpolation precision
and filters numerically degenerate slivers, recording their count and tiny
geometric threshold. The final provenance is finite and exportable.

Independently opened each released NPZ. All four have lead masks
`[1,12,768,1024]`, occurrence masks `[1,15,768,1024]`, lead boxes `[1,12,4]`
and occurrence boxes `[1,15,4]`. Each of the four separate YOLO manifests
contains its one expected sample. This verifies the saved artifact structure
and successful export, not a detector accuracy benchmark.

No remaining release-blocking numerical issue was observed in this review.
The complete final integration suite reported 194 tests, 190 passes and four
CUDA skips. External real-image and clinical generalization validation remain
unperformed. Release is supported by the documented engineering and visual
evidence within that scope.

This is a qualitative comparison of generated examples, not a blinded study
or an empirical estimate of how often users mistake them for photographs.
