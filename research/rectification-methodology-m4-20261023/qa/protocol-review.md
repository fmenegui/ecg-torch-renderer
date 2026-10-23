# M4 independent adversarial protocol review

Reviewer: `m4_adversarial_qa`. Date: 2026-10-23. This review owns only `m4/qa`. It does not change M3, production modules, neural weights or frozen scores. The finite coverage matrix contains 48 failure families. It is not a claim to cover every possible ECG input.

## Decision before new component results

The proposed shared projective normalization, regularized continuous residual and input-only candidate selection are reasonable development hypotheses. Streaming integration and validated atomic image export address reproduced implementation defects. None can currently be promoted as a generally superior rectifier. Approval of a component property, native-image performance, clinical signal fidelity and commercial-product superiority are different decisions with different evidence.

The following are blocking conditions for an upgrade in the claim:

1. Scale, lattice identity, origin, orientation and model-fit uncertainty must remain distinct. Held-out prediction of contaminated but self-consistent integer IDs does not validate those IDs. Perfect cross-validation can coexist with a factor-of-two physical scale error.
2. A continuous map with positive local Jacobians is not by itself globally injective. Require an applicable topology argument with a simple boundary, or independently reject positive-area overlaps and ambiguous preimages. Testing a roundtrip on one selected inverse branch is insufficient.
3. Sampling confidence must never delete observed RGB. Geometry eligibility, actual reconstruction footprint, pixel alpha and epistemic trust need separate semantics and saved arrays. No-grid does not mean no content.
4. Candidate selection must be fixed before reference errors are inspected. Truth is allowed only in the evaluator. A selector that chooses the best physical score per image is an oracle, not an implementable method.
5. A complete-paper mean cannot establish ECG preservation. Thin spikes, decimal points, labels, margins and headers can occupy very little area and disappear without materially changing that mean.
6. M3's compatible Kaggle stratum has one capture from one patient and one backend. It cannot establish superiority on Kaggle's supported task. Our detector's training in the same renderer family also remains a material confound.
7. Dotter/Gridder are unavailable product endpoints in the existing evidence. Patent-inspired geometry is a component proxy and must remain labelled as such.

## What M3 already establishes

The M3 audit verifies 160 scheduled diagnostic outputs on 32 previously inspected synthetic captures from eight patient groups. It establishes that the declared scores, failure accounting, hashes and descriptive summaries are reproducible. It does not turn this consumed panel into a fresh assessment. Our M2 has primary paper G 55.272%, W 15.665%, C 70.937% and R 22.083%. Only 18/32 cases deliver resampled RGB. The native orientation contribution is large, while the M2 versus B0 G difference is 0.252 percentage points. A sampling control independently reproduces complete loss of an observed thin stroke with exact geometry. The initial perception of deletion therefore has a real mechanism without requiring an RGB confidence mask.

M3 also retains negative results. Local composition overfits noisy planes and can worsen wrong coverage near an oblique crease or coherent correspondence fault. More area-integration subpoints do not monotonically improve every objective. A finite 2×2 or 8×8 rule is not a guarantee for arbitrary content, and the input may already have lost information before rectification.

The strongest methodological baseline remains a global homography on a true planar sheet with correct correspondence. A flexible method must first preserve that exactness or show an explicitly measured noise/coverage tradeoff. A more flexible model is not automatically a more accurate one.

## Review of the new component protocols

Geometry protocol hash at first review: `a3ae872d70b34206a8ff809bca10046aa1d3adc81b6c9d9765f50506ac544fe4`. It fixes 18 candidates and a blocked held-out rule before new results. It keeps LS/global first, uses fixed RANSAC settings, includes unregularized and direct-M2 comparators, and retains coherent alias/origin controls. It explicitly treats supplied IDs and 5 mm pitch as assumptions. These are good safeguards against gold winner selection.

The four spatial folds are not four independent patients and their standard error is a heuristic complexity rule, not a confidence interval on physical accuracy. Local spatial dependence, contaminated IDs and observation bias persist. Record every fold loss and chosen candidate, as well as uncalibrated disagreement. A residual fit trained after normalization must exclude withheld source coordinates from both stages. Canonical IDs can remain known, but held-out observed coordinates cannot leak into fitting, inverse support construction or scale fitting. The protocol's conservative support intersection avoids claiming unknown holes as observed. It also means this component does not solve full-paper completion.

Sampling protocol hash at first notice: `228329fb3e8f50bcdb39911aa2ab91f3f7613f0ed6411d69ae8bdeff1900043e`. Planned semantics use source pixel centres, explicit white RGB background, premultiplied RGBA with transparent background, finite/inside/reconstruction/trust fractions and a false invalidity flag that overrides a safe technical coordinate. The independent checks will target those promises, rather than just reproducing candidate code. Border/background policy must state whether outside area is padded or renormalized. These yield different energy and color outcomes. A sampled alpha channel cannot substitute for geometric trust.

Geometry-map sampling and source reconstruction are separate approximations. The midpoint rule samples the composed inverse at subpixel positions. Under a nonlinear warp the source footprint is curved, so an affine strip result does not establish accuracy for that footprint. Quadrature order, source reconstruction, output density and geometric uncertainty must all be disclosed. A cheap refinement disagreement is a useful warning but agreement of two finite rules cannot certify all unseen frequencies.

Atomic export should encode to a temporary file in the destination directory, fully load the decoded pixel array and then atomically replace. A failure before replace must preserve the previous valid target. Atomic replacement does not establish power-loss durability unless directory synchronization and its failure states are addressed. The M3 truncated PNG incident is evidence that dimensions-only checks are insufficient.

## Executable next benchmark

The next benchmark should be a locked runner and manifest, not a new average on the same inspected 112-capture family. The following lanes answer different questions and should remain separate.

| Lane | Input available to methods | Purpose | Independent truth | Allowed conclusion |
|---|---|---|---|---|
| A: geometry association conditional | Identical observed nodes, IDs, pitch and explicit support | Compare solver and selector choices | Closed-form maps and independent camera/surface controls | Solver behavior conditional on supplied assumptions |
| B: sampling/export conditional | Identical image and exact inverse | Isolate interpolation, footprint, color and export | Analytic pixel areas or an independent fine reference with a documented bound | Sampling fidelity for stated scenes and resolution |
| C1: native compatible task | Identical fresh raw captures in the fixed 3×4+r1 task | Compare pinned authentic Kaggle, npj and our endpoints on common supported input | Independent calibrated geometry, supplied waveform and source-annotated objects | Distribution-specific endpoint comparison |
| C2: native broad task | Fresh layouts and capture sources declared before inference | Measure operational utility across the target application | Same independent truth contract | Deployment-distribution coverage and failure description |
| D: real capture transfer | Independent paper/printer/camera pipeline outside all development renderers | Test renderer training advantage and real observation effects | Calibrated capture rig or independently metrologized grid/page | External geometric generalization for represented acquisition conditions |
| E: downstream fidelity | Same fixed extractor on each rectified output plus an original-image control | Test mV/time/lead and tiny-feature retention | Untouched supplied signal, printed windows and object masks | Downstream evidence, not clinical diagnosis superiority |

For C1, use a patient-level allocation before inference and preserve equal counts by printer/camera/source family. A practical first stage is 48 independent patient records, three preassigned captures each, including one ordinary scan, one moderate photo and one adverse photo. All must be generated or acquired after the complete candidate set is frozen. At least two external renderer or export families are needed, and the real subset must use multiple printers and cameras with their identities retained. If these data or permissions are unavailable, this lane remains not executed. The existing one compatible case cannot be reused as a substitute.

The number 48 is an executable engineering starting point, not a power calculation or a license to claim rare-event safety. For a simple independent patient-level zero-failure bound, at least 149 patients with zero events are needed for the one-sided 95% binomial upper bound to be below 2%. Dependence across captures, acquisition clusters, multiplicity and nonzero events change the design. A formal superiority study must choose its meaningful effect and sample size before outcomes, with a paired patient-level analysis and multiplicity policy for the actual comparator list.

Use an untouched development pool from different records for all detector thresholds, pitch hypotheses, fold sizes, clipping losses, sampler defaults and selector rules. Freeze candidate source, weights, source/capture manifest, support assumptions, queries, budgets and metric implementation. Keep truth in separate evaluator files inaccessible to inference. Materialize a deterministic schedule containing every method/case combination. A failure, timeout, original fallback, invalid field or unsupported case remains in its declared denominator. No case replacement and no run retries selected by quality. Recovery after a corrupted artifact needs an incident record, preserved bytes and no recomputed score.

Each authentic comparator retains its native preprocessing and checkpoints. Supported-layout strata must be determined from official pinned assumptions before inference. More than one Kaggle solution may implement the same geometric idea, so report exact implementation identities rather than counting correlated presets as independent confirmations. If an adaptation is needed, preserve the authentic arm and label the adaptation separately. Unknown original training overlap must be recorded rather than assumed absent. Native end-to-end performance and common-observation solver performance should not be mixed into a single ranking.

For real geometry truth, an ordinary ECG grid is both a method cue and a potentially biased ruler. Use an independently metrologized sheet, external fiducials or calibrated camera/surface acquisition. Ensure visible fiducials do not hand one method privileged inference cues. A clean scan of the same print is useful for material correspondence only if scanner warp, calibration and registration uncertainty are characterized. Fitted homography to the same nodes under test is not independent gold for a nonlinear solver. Hidden/occluded content must be reported as unidentifiable, not imputed to become ground truth for visible recovery.

## Required metrics and acceptance gates

No single winner score should replace the following gates. Predeclare engineering tolerances and keep them distinct from clinical acceptability.

| Gate | Measurement | Blocking failure |
|---|---|---|
| Identity and units | Pitch/axis/orientation/lead/time convention, native origin and ambiguity | Silent unit, reflection, permutation or coherent lattice association error |
| Geometry | G/W/C/R, p95/p99/max, patient failures, actual source footprint | Accepted nonfinite, fold, overlap, ambiguous inverse, hidden failed case |
| Planar preservation | Exact clean homography and noisy-plane baseline | Flexible candidate chosen with added clean-plane bias or undisclosed noise regression |
| Full output | Actual resampling status, original fallback, partial coordinates and bytes retained | Partial map or original archive counted as full rectification |
| Content | Object-wise spike/point/text survival, contrast and dark-area bias | Tiny observed element lost but hidden by image mean or confidence mask |
| Waveform | Common-extractor mV/time/lead errors against supplied original signal | Lead/time/polarity corruption or a changed extractor presented as rectifier gain |
| Selective prediction | Coverage-risk relation plus fixed-domain failures | Lower risk obtained by suppressing difficult regions without coverage disclosure |
| Export | Full decode, pixels/mode/hash, provenance, injected precommit failure | Partial file accepted or previous valid target lost before commit |
| Runtime | Same CPU/device, cold/warm scopes, RSS and output budget | Cached map application compared to full native pipeline as a speedup |
| External validity | Patient/source/device counts and training overlap | Same-renderer improvement called universal, clinical or commercial superiority |

Query measures also need separation. M3 samples uniform source material-support pixels. This is valid for the stated image domain but overweights some material regions under perspective. Add a separately predeclared uniform canonical material-area query set for physical geometry and keep the original source-pixel set for operational image coverage. Neither is a replacement for object-wise signal/text masks. Report grid, visible full paper, no-grid margins/header and occluded fraction separately. Use the same truth-derived domains for every method, independent of the predicted confidence mask.

The 0.5 mm M3 threshold is an engineering comparison threshold. Its time or voltage meaning depends on actual paper speed and gain. A translation-aligned geometry score removes chart-origin ambiguity but does not prove an automatically correct time origin or voltage baseline. Never use per-lead, per-component or per-domain gold registration to hide such failures.

## Independent QA execution commitment

The independent test specification was saved before reading new component outputs as `independent-test-spec.json`, SHA256 `95d5ef1643fb605b894f96a30d84d8e4f1d313aa82cbcbd0300b478563023797`. The initial 48-family coverage matrix SHA256 is `842434b32123c03e2365cc6586d3b1544e4c6b4387544babeed7be299f7fbeea`.

The tests will use independent affine/projective formulas, malformed and globally overlapping inputs, direct scalar interpolation, explicit half-open pixel geometry and injected export failures. They will retain failures, distinguish mathematical properties from finite empirical controls, and record component source hashes. They will not rerun neural inference, edit frozen M3, tune new candidates on truth or claim to have attacked every input. The final review will separately list proven conditional properties, executed component results, unidentifiable cases and external clinical/product evidence that remains unavailable.
