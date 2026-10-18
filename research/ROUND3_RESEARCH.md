# Round 3: physical paper damage and realism review

Reviewed 2026-10-18. This extends `ROUND2_DOCUMENT_RESEARCH.md` with mechanisms for
creases, crumpling, cuts, missing material, stains and thermal-paper condition.
Primary sources were inspected online. This research task did not import external
code, assets or datasets, run a mechanics solver, or establish photographic
indistinguishability. Proposed gates are engineering criteria, not measured
results or medical standards. Production implementation and test results belong
to the other round 3 reports.

## Main finding

The largest improvement comes from **coupling causes**, rather than increasing
independent noise strengths. A folded page needs corresponding grid deformation,
normal change, illumination and shadow. A tear removes paper, grid, labels and
trace together. A thermal stain changes print chemistry differently from an
ordinary brown liquid deposit. A camera pose changes paper and damage together.

The round 2 path-traced gallery was visually inspected as a baseline. Its single
fold and curl are coherent but unusually clean and uniform, the sheet silhouette
is pristine, and the same dark tabletop dominates all three examples. Those are
observations of the gallery, not evidence from a blinded realism study. Neither
more path samples nor stronger sepia alone addresses these cues.

## Verified primary sources and what they support

| Source | Mechanism supported | Engineering consequence |
| --- | --- | --- |
| [Grinspun et al., Discrete Shells, SCA 2003](https://multires.caltech.edu/pubs/ds.pdf), [institutional record](https://authors.library.caltech.edu/records/nagsj-q9097), DOI 10.2312/SCA03/062-067 | Triangle-mesh membrane and bending energies. Rest dihedral changes model plastic bending. | A permanent crease is a reference-shape change, not an extra painted stripe. Distinguish shell mechanics from a prescribed height field. |
| [Narain, Pfaff and O’Brien, Folding and Crumpling Adaptive Sheets, TOG 2013](https://dl.acm.org/doi/10.1145/2461912.2462010), [author page](https://objf.ai/papers/Narain-FCA-2013-07/) | Plastic sheet simulation with adaptive remeshing aligned to sharp folds and a plastic embedding. | Resolve narrow hinges and flatter facets without pretending a uniformly coarse mesh is a mechanics solution. |
| [Cerda et al., Conical dislocations in crumpling, Nature 1999](https://www.nature.com/articles/43395), DOI 10.1038/43395 | Deformation localizes around ridges and point-like peaks between nearly planar or cylindrically bent regions. | Smooth everywhere-curved noise is a poor visual model for an opened crumpled page. |
| [Pfaff et al., Adaptive tearing and cracking of thin sheets, SIGGRAPH 2014](https://tobiaspfaff.com/publication/pfaff-2014-adaptive/) | Research reference for evolving cracks in deformable sheets. | True tear propagation is a separate mechanics task. A removed polygon should be described as a cutout approximation. |
| [Augraphy Folding documentation](https://augraphy.readthedocs.io/en/latest/doc/source/augmentations/folding.html) | Two perspective transformations emulate a fold. Masks and keypoints are supported, while boxes only move their start or end point. | Useful augmentation comparator. Recompute our boxes from final per-instance support instead of copying this box update rule. |
| [Journet et al., DocCreator, J. Imaging 2017](https://doi.org/10.3390/jimaging3040062), [author repository](https://github.com/DocCreator/DocCreator) | Ground-truthed document degradation, holes derived from real defect examples and deformation using captured geometry. | Measured defect banks can improve morphology more than a larger ideal-shape catalog. Repository LGPL-3.0 is not a license to silently copy code into MIT files. |
| [UVDoc author dataset](https://github.com/tanguymagne/UVDoc-Dataset), DOI 10.1145/3610548.3618174 | Captured deformed white sheets with UV, segmentation and 2D/3D grids, later combined with document textures. | Still the strongest practical measured-geometry route. Keep captures, albedo and illumination semantics separate and split by geometry identity. |
| [Deegan et al., Capillary flow as the cause of ring stains from dried liquid drops, Nature 1997](https://doi.org/10.1038/39827) | Pinned contact lines and evaporation-driven outward transport concentrate deposits near a boundary. | A variable rim and lower-density interior are more defensible than one uniformly filled circle. This is not a universal paper-stain simulator. |
| [González-Gutiérrez et al., Pattern formation by droplet evaporation and imbibition in watercolor paintings](https://arxiv.org/abs/1909.09098) | Experiments show borders, gradients and other patterns varying with pigment concentration and paper humidity. | Vary wetness history and deposition morphology. One perfect coffee ring for every liquid is unrealistic. |
| [Bloch et al., Anisotropy and Fiber Orientation: A Key Player in the Lateral Imbibition of Cellulose Paper, Fibers 2024](https://www.mdpi.com/2079-6439/12/7/56) | Experimental in-plane imbibition depends on paper structure and fiber orientation. | Anisotropic feathering is a useful approximation, without claiming the same measured parameters for coated ECG paper. Publisher page rate-limited during full-page access, search-index abstract was available. |
| [Koehler official thermal-paper durability note, April 2022](https://www.koehlerpaper.com/media/docs/en/product-information/thermo/technische-infos/TH-Haltbarkeit-von-Koehler-Thermopapieren_GB.pdf), [Jujo official storing guide](https://www.jujothermal.com/technical-guide/correct-storing/) | Paper-grade-dependent sensitivity to light, heat, humidity, plasticizers and chemicals. Jujo also documents image development from frictional heat and image fading from sweat. | Separate background darkening, contrast loss and local handling marks. Do not convert an arbitrary severity slider into years of ECG-paper aging. |
| [Duan, Micro-fading Analysis of Historical Thermal Imaging Media, Microscopy and Microanalysis 2024](https://doi.org/10.1093/mam/ozae044.064) | Different historical image-forming chemistries have different light-stability responses. | Evidence against a universal aging law. Historical thermal copying media are not a calibration dataset for contemporary ECG thermal paper. |

No source above establishes that this repository is the best ECG renderer or
that a procedural image can be guaranteed 100% realistic. Literature examples
explain mechanisms. They do not supply the prevalence or severity distribution
for hospital ECGs.

## Implementable geometry decisions

### Fold versus opened crumple

A single handled fold should have unequal panels, a finite-width hinge and some
residual curvature. Its orientation, location, angle and ridge width can vary.
An opened crumple needs a connected, irregular ridge network with branches and
vertices, flatter intervening facets and several weaker residual wrinkles.
Avoid identical, infinite, parallel sine waves across the whole sheet.

For the torch path, a bounded nonperiodic network of short roof-like creases is
a defensible approximation. Make ridge endpoints taper and permit mountain and
valley signs. Compute normals from the same differentiable geometry used by
projection. Report the local metric strain and inverse-map residual rather than
assuming an added height field is inextensible. A graph height field cannot
represent an overhang or a fully closed fold, so expose that limitation.

For offline 3D, progressively rotated panels can preserve length along selected
directions. General crumpling still requires strain checks and self-collision
handling. A textured arbitrary triangle mesh rendered with correct light
transport is physically coherent imaging of that mesh, not evidence that the
mesh is an equilibrium shape of paper.

Proposed checks: compare original and deformed edge lengths, identify inverted
or degenerate triangles, inspect Gaussian-curvature concentration, and verify
that uniform albedo shading changes correctly when light direction changes.
There should be no unexplained printed dark stripe that stays fixed when the
illumination reverses. Crease wear may remain fixed because it is a distinct
material effect, and must be labeled as such.

### Perspective, clipping and missing regions

Use a camera pose or a well-defined planar homography followed by the documented
nonplanar projection. Derive all target fields from that same geometry. Camera
cropping, another sheet covering the page, a removed piece and a pale stain may
look superficially similar but have different ownership and supervision.

Maintain these distinctions:

- Missing substrate exposes the table or whatever surface is behind the hole.
- An opaque object removes visible target support but does not delete the source
  paper from its amodal geometry.
- Image cropping removes support outside the frame.
- Fading reduces readability while the source trace can remain geometrically
  present. A geometric mask should not silently become a contrast threshold.

Use local material coordinates for tears and stains, then project once. Verify
that an asymmetric defect follows a page rotation and never stays fixed to the
camera. Recompute each printed-instance box from its final visible mask. The
semantic box of a repeated lead can enclose widely separated fragments, so keep
per-instance YOLO boxes in addition to the twelve semantic classes.

## Implementable damage and material decisions

| Phenomenon | Better bounded approximation | Artificial cue to reject | Proposed criterion |
| --- | --- | --- | --- |
| Scissor cut | Straight or gently kinked polygonal boundary with low roughness and a defined removed side | Jagged fibers on every clean cut | Boundary slope stable at native scale, correct background ownership |
| Torn edge or missing corner | Nonuniform boundary with several spatial scales and sparse short fibers | One repeated sawtooth frequency, symmetric triangular notch everywhere | No stable narrow spectral peak across many seeds, thin edge confined to a physical-scale band |
| Punch hole | Manufacturing-sized circle or ellipse after projection, possible repeated alignment | All damage types forced into ideal ellipses | Physical dimensions recorded, target void matches rendered void |
| Open slit | Narrow material removal, possibly revealing a displaced lip in 3D | Thick universal dark stroke mistaken for a hole | Visibility changes with width and pose, no trace across removed support |
| Coffee-like residue | Irregular perimeter, uneven deposit width and density, low interior pigment and occasional overlapping drops | Identical circular ring with uniform brown alpha | Higher mean deposit near rim than interior for ring mode, irregularity varies with seed |
| Diffuse liquid mark | Anisotropic feathering, soft interior modulation and optional drying fronts | One hard Gaussian blob type for every liquid | Blur/roughness and density vary independently within bounded ranges |
| Thermal fade | Contrast attenuation in print formation, optionally spatially correlated with exposure or handling | Uniform sepia filter with pristine black traces | Signal contrast decreases locally without moving trace centerlines |
| Thermal activation | Local background darkening with plausible patch or scratch morphology | Ordinary coffee coloration used for a heat mark | Darkened unprinted margins and affected printed regions share material coordinates |
| Handling/abrasion | Sparse local wear near folds and exposed edges, independently optional | Entire page coated in equal-strength high-frequency speckle | Real-scale crop inspection and optional measured-bank comparison |
| Paper aging | Low-frequency nonuniform base reflectance change, distinct from illumination and print fade | Dark vignette at every edge presented as years of aging | Constant light reveals persistent albedo change, no calendar-age claim |

These are design inferences informed by the cited mechanisms. They are not
parameter fits to ECG photographs. Small fibers and edge thickness should fade
through antialiasing when below a pixel, rather than become a resolution-sized
white or black halo. Millimeters or fractions of the physical page must define
feature scale before rasterization.

### Compose mechanisms in causal order

One useful order is source waveform and print, substrate condition and removal,
physical geometry, illumination/occlusion, optics/sensor and encoding/compression.
Print degradation needs its own place in this chain. Optical blur should affect
all image components consistently, while geometric ground truth retains its
explicitly documented coverage semantics.

Correlate events through a small number of scenario families. A stored folded
thermal print can combine residual creases, local contact fade and uneven base
color. A fresh phone capture can be clean but oblique, with mild curl, hand
shadow and optical blur. A scanner capture can flatten most geometry while
retaining crease material marks and edge damage. Do not make every severe
phenomenon appear in every image. Relative frequencies still need calibration
from real deployment images.

## What this round can and cannot establish

Deterministic replay, finite geometry, mask alignment, correct material removal
and consistent light transport are directly testable software properties. They
are necessary for reliable training data and should gate release.

Photographic realism, scenario prevalence and improvement on held-out clinical
captures are empirical properties. A larger gallery supports review but does
not establish them. No external real-paper capture bank was evaluated by this
research task. No defect-mask bank or external source code was copied.

The next strong experiment remains paired captures: render the same canonical
ECG used for physical printing, apply controlled handling to actual sheets, and
photograph/scan before and after. Reserve paper brands, operators, devices,
lighting settings and recording IDs for held-out evaluation. Estimate parameter
distributions on the calibration subset only. Evaluate digitization error plus
per-lead detection and segmentation, so an attractive image cannot hide damage
to supervision or physiological content.

## High-resolution visual review rubric

Use fixed source records and a fixed seed manifest. Review full pages and 1:1
native-resolution crops before resizing. Include at least clean, mild and severe
cases for each scenario family, plus combinations and complete lead removal.
For every image, inspect RGB, paper support, twelve lead masks and per-instance
boxes side by side.

Score each criterion 0 for clear failure, 1 for visible approximation and 2 for
credible appearance in the inspected view. Retain the individual scores rather
than hiding a severe target defect in an average.

| Criterion | Full-page inspection | Native-resolution crop inspection | Automatic companion |
| --- | --- | --- | --- |
| Sheet geometry | Plausible facet layout, no rubber waves or repeated fold template | Finite hinge width and coherent grid kink | Strain, mesh area and inverse residual |
| Surface lighting | One coherent scene light and support shadow | Highlight/shadow changes follow normals | Flat-albedo light reversal fixture |
| Edges and voids | Distinct cut, tear, hole and crop morphology | No staircase at normal viewing size, no thick universal halo | Final support and ray-hit ownership |
| Stains | Irregular plausible deposit within substrate | Nonuniform rim and feathering, no tiled texture | Boundary statistics across seeds |
| Aging and print | Trace/grid/base respond according to selected material model | No independent white ghosts, trace centerlines preserved | Source contrast and coordinate checks |
| Capture optics | Perspective and blur fit the capture family | No mismatched sharpness at damage boundary | UV fiducials and PSF support checks |
| Supervision | Repeated strips retain class and instance identity | Masks follow visible trace coverage through defects | Independent mask-to-box oracle |
| Diversity | Similar causes vary coherently across seeds | No repeated microtexture or fixed-frequency edge pattern | Cross-seed autocorrelation diagnostic |

A reviewer should fail any image whose target ownership is wrong even if the
RGB is attractive. Render convergence and geometry correctness do not substitute
for comparison with real photos. **Held-out photographic validation remains
absent from this research review, so no 100% realism claim is justified.**
