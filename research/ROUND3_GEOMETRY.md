# Round 3 material geometry review

## Implemented changes

The real-time paper model now places finite crease networks away from the
sheet center. Four independent parent ridges can each carry a smaller branch.
The branch begins at a displaced point on its parent's material-space line,
inherits its scale and polarity, and varies its angle, width and length.
Each ridge has unequal transverse facets and unequal longitudinal extents.
The support boundary blends smoothly into the surrounding facet. This avoids
introducing a second hard ridge at the procedural support boundary.

Broad sinusoidal deformation in `crumpled` was reduced. The main visible shape
comes from localized folds. Small secondary buckles follow those same crease
directions. Randomness changes their phase rather than adding an isotropic
height field. Flat and developable surfaces receive none of this detail.

The single-fold mode has mild distributed angular drift. Accordion panel
lengths vary through a monotonic remapping of fold phase. Lifted corners now
combine a narrow hinge and a wider curvature transition. These modes still
integrate unit tangents before projection. The curl and rolled formulas were
preserved.

There are no additional public parameters or random draws. Existing sampled
parameters and the existing detail-noise draw are sufficient. The integrator
must bump the `paper_model` fingerprint because replay under the previous
geometry model should not silently produce a different surface.

## Independent verification

Command:

```text
python -m unittest tests.test_paper tests.test_paper_r3 tests.test_physical_scene -v
```

Result: **22 tests executed, 21 passed, 1 skipped because CUDA is unavailable**.
The existing eager-versus-compiled scene comparison passed, including the
generator-state comparison, inverse field and trace supervision.

Four additional regressions check:

- The measured maximum incremental ridge height occurs at the prescribed
  off-center material junction. The branch maximum occurs at the displaced
  point on the parent. This check does not use the inverse warp as its oracle.
- Accordion panel intervals are unequal. Direct neighboring mesh-edge lengths
  constrain stretch separately from the implementation's returned metric.
- Changing the detail-noise realization cannot displace flat, curl, fold,
  accordion, rolled or lifted-corner meshes.
- A batch of 64 surfaces at maximum strength has bounded direct horizontal,
  vertical and diagonal edge-length errors. Diagonal lengths also constrain
  shear. No metric helper is used as the reference for this check.

For the 64-case audit, the maximum per-surface mean absolute metric error was
0.0018604. The maximum mode-wide 99th percentile was 0.0159792. Existing gates
are a mean below 0.004 and a 99th percentile below 0.025. The single fixed
eight-mode legacy physical audit also passed unchanged.

| Surface | Mean absolute metric | 99th percentile |
| --- | ---: | ---: |
| Flat | 0.00000057 | 0.00000191 |
| Curl | 0.00001549 | 0.00003946 |
| Fold | 0.00025001 | 0.00759575 |
| Accordion | 0.00109278 | 0.01597920 |
| Crumpled | 0.00059440 | 0.00867314 |
| Wrinkled | 0.00016036 | 0.00199374 |
| Rolled | 0.00003207 | 0.00026786 |
| Corner | 0.00001819 | 0.00051427 |

A construction-free CPU geometry timing measured a median 153.3 ms for 64
surfaces at mesh width 96, two PyTorch threads, PyTorch 2.14.1+cpu. This is a
component measurement under a shared workload, not a full renderer benchmark
or a GPU claim. Raw audit and test output are in the local integration build
directory `build/round3-geometry/`.

## Visual review and limits

The isolated Lambertian before/after diagnostic in
`build/round3-geometry/geometry-before-after.jpg` was inspected. It shows
localized asymmetric ridge junctions and less broad rubber-like curvature.
It deliberately removes ECG print and aging so geometry can be judged alone.
The scene integrator should inspect the final adverse-capture gallery too.

This remains a conservative 2.5D prescribed height-field model with a
small-slope shortening approximation. It does not solve elastic shells,
plastic rest angles, arbitrary self-contact, buckling equilibria or deeply
crushed paper. Local crease junctions can retain residual metric error. An
opened accordion is still translationally developable along its fold axis.
This result does not establish indistinguishability from photographed paper.

Physical motivation follows the distinction between facets, localized ridges
and vertices emphasized in [Narain, Pfaff and O'Brien, 2013,
*Folding and Crumpling Adaptive Sheets*](https://dl.acm.org/doi/10.1145/2461912.2462010)
and the shell discretization literature, including
[Grinspun et al., *Discrete Shells*](https://multires.caltech.edu/pubs/ds.pdf).
Those works motivate the geometry design. This implementation is not an
implementation of their dynamics solvers.
