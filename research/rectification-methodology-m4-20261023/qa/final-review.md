# M4 independent adversarial review

Reviewer: `m4_adversarial_qa`. Date: 2026-10-23.

**Decision: accept the reviewed components as conditional development prototypes with retained failures. Do not promote M4 as a complete superior ECG rectifier. One blocking topology defect was reproduced, preserved and corrected. Fixed finite quadrature still has an independently reproduced complete-stroke-loss case. A separate restricted exact-area control repairs that case under its stated source model.**

No neural pipeline, frozen M3 file, reference waveform or production module was changed by this QA work. There are no new real clinical captures or commercial product executions in this review.

## Evidence actually reviewed

| Evidence | Independent result | What it establishes |
|---|---:|---|
| Component adversarial suite | 21 tests pass | Independent scalar RGB/RGBA reconstruction, trust separation, borders, nonfinite/extreme coordinates, tile invariance, bounded explicit coordinate batches, PNG failure injection, clean homography, holes, horizon, malformed IDs, global wrap rejection and C0 limits |
| Saved geometry experiment audit | 6,279 checks pass | All 76 primary outputs and 342 candidate outputs across 19 scenes, frozen selection rules, decision seals, analytic query formulas, scores, common support and disagreement recomputed |
| Saved sampling/replay audit | 23,819 checks pass | Schedule and counter audit for 3,360 archived comparisons, 12 quality summaries independently recomputed and 192 selected outputs rerun from direct affine coordinates |
| Separate exact-axis-area suite | 5 tests pass | 160 output pixels checked against independent source-cell rectangle clipping, RGB/RGBA, borders, anisotropy/flips, tile invariance, rejection of shear/singular maps and unresolved tiny footprints |
| Final-source cost and full replay receipts | 67 checks pass | Final sampler identity, three-repeat medians, original/final cost hashes and the complete 3,360-output final replay audited without remeasuring runtime or RSS |
| Prior M3 results | Read-only continuity review | Retrospective synthetic diagnostic results remain unchanged, including one compatible Kaggle case and known renderer-family training advantage |
| Identifiability witnesses | Source/report interpretation reviewed | Finite mathematical/contract examples, not a new detector or photographic endpoint evaluation |

The 26 test methods are not 26 patient cases. The 30,098 saved-evidence checks are not independent clinical observations. Hundreds of candidates and thousands of pixel queries are not additional independent patients.

The coverage matrix lists 48 failure families. Its explicit current M4 states are **21 with finite component tests, 13 partially exercised, 6 represented by reviewed constructive witnesses and 8 not executed in M4**. A finite test does not exhaust its entire family, and reproducing a failure counts as an executed attack rather than a successful correction. None of these counts implies all possible inputs have been attacked. The initial pre-result matrix remains in `coverage-matrix-initial.json` and the updated matrix in `coverage-matrix.json`.

## Blocking topology defect and correction

The initial geometry guard rejected mixed triangle orientation and positive-area overlap. I independently constructed a rectangular strip with two radial rows, r=1 and r=2, and 25 angular columns spanning theta=0 through 2π inclusive. It becomes an annulus whose two distinct boundary endpoints coincide in the source. All 48 triangles have the same orientation, and there is no positive-area overlap.

The initial guard accepted it. Forward evaluation at source `[1.5,0]` returned chart `[0,2.5]` as an ordinary returned coordinate, although the same source point also represents chart `[120,2.5]`. The roundtrip rejected one inverse branch but did not make forward evaluation unambiguous. The preserved receipt and source snapshot are:

- `receipts/non-simple-boundary-before-fix.json`
- `snapshots/geometry_selector-pre-boundary-review.py`

The failed source SHA256 is `42737558261ca6f5f87a04be8e208dc5a779918a487f39b11d1e3f76aaf37239`. This is concrete evidence that positive local orientation and no positive-area overlap do not alone settle boundary injectivity.

Guard-v2 additionally rejects distinct referenced vertices that coincide and touching/crossing nonincident boundary edges. The independent suite now rejects both the one-turn boundary-contact case and a two-turn positive-area wrap. Independent clean-plane, hole, exterior and 173 shared-edge controls remain valid. The 173 two-sided edge limits agree with their shared targets to the declared tolerance, with maximum two-sided epsilon-limit difference 3.727×10⁻⁷ mm at epsilon 10⁻⁷. This finite numerical check supports the stated C0 construction. It is not a theorem about every floating-point mesh or a physical certificate for a photographed surface.

The owner retained v1 and reran the same frozen 19 scenes. My audit of v2 independently recomputed all 76 outputs and 342 candidates. The reported decisions, coordinates, masks and scores remain unchanged by the guard repair. Candidate families and selection hyperparameters were not tuned from physical scores after this defect. Full v1/v2 archive parity is additionally recorded by the owner's retest. My approval covers the reviewed dense formulation and guard-v2. The separate sparse optimization and its performance results were not independently rerun here.

## Geometry improvements and failures that remain

The selector receives only observed source points, supplied integer IDs, allowed cells and assumed pitch. Its interface does not receive gold. The saved decisions are sealed before physical query generation, and the 18-candidate ordering plus blocked-fold rule reproduce independently. Source inspection supports this isolation for the reviewed implementation. It is not a formal noninterference proof for arbitrary future globals or callbacks.

The results support retaining a global H candidate for planar denoising and composing the shared continuous residual with H. On the high-noise plane, RMSE changes from 0.26844 mm in direct M2 to 0.05053 mm in the selected candidate. On the smooth noisy scene it changes from 0.20269 to 0.15546 mm. Clean projective scenes have selected errors around 10⁻⁸ mm. These are analytical component outcomes conditional on the supplied observations and IDs.

There is no universal dominance:

- The clean oblique crease worsens from direct-M2 RMSE 0.09217 to 0.10748 mm. W rises from 0.2686% to 1.2207%. This blocks a claim that regularization is always the right decision.
- The planar outlier scene is rejected completely. C=0 means no geometry delivered, with R undefined. Zero W on this scene is not success.
- Coherent row and half-sheet faults retain W of 72.75% and 79.54%. Selection does not validate association.
- The coherent 10 mm versus assumed 5 mm pitch alias produces W=100% with RMSE 53.94 mm despite near-zero internal CV loss. It blocks a physical scale certificate based on CV, cycles, smoothness or agreement.
- The shifted absolute-origin control has 11.18 mm raw RMSE. This demonstrates a gauge ambiguity, not evidence that a relative rectifier can infer an unobserved origin. A translation-only diagnostic removes this constant origin shift but cannot remove the pitch alias.
- Physical exterior queries can be returned when noisy or corrupted observed support extends outside the true paper. The coherent half fault returns 52 of 256 physical exterior queries. Being inside the inferred observed mesh is not independent evidence of physical-paper membership.

No-grid, gap and margin behavior remains a distinction between geometric support and visible content. The component preserves explicit holes rather than filling them. It does not implement a reliable full-paper completion model, and it does not train or validate a new frontend.

## Sampling, export and the retained disappearance attack

The new streaming sampler passes 12 RGB/RGBA output comparisons against an independently written scalar reconstructor without a cv2 reference call. Actual maximum uint8 difference is zero. True, false and mixed geometry trust produce exactly the same RGB and alpha bytes. Nonfinite/extreme coordinates contribute explicit background and never borrow the valid technical coordinate `[0,0]`. Border tests independently check the half-open source-cell domain and non-renormalized outside fill. RGBA premultiplication excludes hidden transparent color. Source reconstruction support, finite geometry, source-cell support and trust remain separate evidence.

Tile tests cover multiple partitions, small/prime budgets and a nonlinear inverse. They prove parity for those exercised partitions, not nonlinear integration accuracy. A 37×53, n=8 streaming run sends at most 31 points per callback and owns at most 1,736 explicit coordinate-array bytes. The O(tile) claim excludes the source copy, a collected output, retained tiles, allocator internals and callback caches. It is not a bound on total process memory.

The 3,360 archived ROI comparisons retain the M3 reconstruction numerics. I independently audit their schedule/counters and rerun 192 selected outputs from direct affine formulas with two tile budgets. All selected channel values agree exactly. The 12 archived quality summaries also recompute. This is parity on the 41×41 interior ROI, not proof of full-frame or nonlinear photo fidelity.

I independently reproduced a decisive negative control. A white 65×65 source contains a black one-pixel vertical column at x=32. A single output pixel uses the exact affine inverse `source=32+32*physical`, with spacing 1. The 2×2, 4×4 and 8×8 rules all return `[255,255,255]`, erasing the observed trace. The exact mean of its bilinear tent reconstruction over the footprint is **247.03125**. The source-cell constant reconstruction has the same mean on this particular footprint. Thus agreement of 2/4/8 rules does not certify preservation, and a fixed n is not a universal sufficiency rule.

The separate exact-area control was developed after that attack and remains labelled failure-directed development. It only accepts a diagonal nonsingular affine inverse. It integrates the declared constant source-cell reconstruction by exact rectangle overlap up to floating-point arithmetic. Any off-diagonal shear or rotation is rejected, including tiny nonzero entries. The independent reference clips each individual source pixel cell, without prefix sums. Eight RGB/RGBA images comprising 160 output pixels agree in uint8, with maximum encoded-mean discrepancy 1.296×10⁻¹¹. The 32:1 case returns the correct mean 247.03125 and uint8 247. The restricted positive control does not establish the right reconstruction model for unknown continuous ECG ink, and it does not solve general projective or nonlinear footprints.

A general sampler still needs a tested footprint-aware anisotropic integration strategy, including support boundaries and an explicit approximation budget. Increasing n or comparing finite rules is a diagnostic, not a physical-error certificate. Original archiving remains useful but does not make a damaged rectified raster faithful.

The PNG writer passes full decode, exact pixels/mode/hash and injected encoder, truncated-temporary and replacement failures. Precommit errors preserve the previous valid target and remove temporary artifacts. Default publication refuses an existing target, guarded replacement rechecks the expected digest, and directory synchronization is explicitly reported. These tests do not promise immunity to later storage corruption or an external writer ignoring cooperative locks.

## Resource evidence

The saved CPU cost comparison includes conversion, coordinate evaluation, reconstruction and collection. Its medians and output hashes were audited, while peak RSS was not independently remeasured. The preserved initial checkpoint is `m4/sampling/results/benchmark.json`. At n=8 on 1024×768 output it reports cached sampling at 2.692 s and about 941 MB process peak RSS, versus tiled sampling at 5.404 s and about 81 MB.

The final-source checkpoint is separately retained at `m4/sampling/results/final-cost/benchmark.json`. It binds the approved sampler SHA256 and reports n=8 cached sampling at **2.185 s and 941,293,568 bytes**, versus tiled sampling at **3.813 s and 81,547,264 bytes**. The 58-check final-checkpoint audit verifies all six three-repeat medians, exact output hashes against the initial checkpoint, source identity, throughput arithmetic and coordinate/cache formulas. The initial measurements remain preserved. Shared-host variability and separate checkpoints prohibit treating their difference as a quality or speed improvement. Both show a clear memory/latency tradeoff, with streaming slower than the cache on this workload.

The 16-million-pixel old-cache value of 8.192 GB is an allocation formula, not a completed 16M-frame measurement. All measurements remain shared-host CPU evidence, with no GPU or full-neural-pipeline speed claim. The integrator's complete final-source replay subsequently finished at `m4/integration/final-sampling-replay/parity.json`. Its 3,360 comparisons, counters, zero mismatches, 12 quality summaries and component hashes exactly match the original replay. The final-checkpoint audit now passes 67 checks in `receipts/final-checkpoints-audit.json`. This strengthens final source/measurement binding. The independent 192-output direct-affine rerun remains separate evidence and was not repeated.

## Approved component identities

| Source | SHA256 |
|---|---|
| `m4/geometry/geometry_selector.py` | `0ce6062797c5e25dd9ce3d7b26a6a9fabe7779ca72b5ea4e55fcda564bed9e87` |
| `m4/sampling/streaming_sampler.py` | `b9a7a732533090271455d42fdc3b4784bf299e14f0b4fc82c564fadccc31f503` |
| `m4/sampling/pngio.py` | `816c319d8bdd47ef82aff7057a4461d7b2b3249ce95d60a7a7fa3b93b3fd38b1` |
| `m4/sampling/axis_aligned_area.py` | `d7b347d7733d91530babf518ab2987c2558b4aebed3b4f37773f6b8fee49cc26` |

An approved identity means the finite reviewed contracts and tests pass in their declared domain. It is not blanket approval of future modifications, a full product or an unconditional mm claim.

## Remaining comparison gap and next gate

The next experiment must separate matched-observation geometry from authentic native frontends, and sampling from both. The executable design in `protocol-review.md` preserves the fixed C1 compatible task, a broader operational cohort, external renderer/real capture transfer, independent metrology and a fixed common waveform extractor. Every scheduled failure and original fallback stays in the denominator. Candidate code, weights, settings, information supplied at inference and scoring domains must be locked before fresh outcomes. No per-image gold winner, scale rescue, D4 rescue or success-selected comparator subset is permitted.

The current one-case Kaggle-compatible stratum and our same-renderer detector training advantage remain unresolved. No new external real geometric gold, common waveform-extractor evaluation, clinically annotated pacing-spike/text-retention cohort or authorized Dotter/Gridder execution exists here. A patent-derived primitive remains distinct from a commercial product. These gaps block claims of superiority to all Kaggle solutions, npj Digital Medicine or Powerful Medical, and block clinical or universal-best claims.

Unidentifiable targets must retain their hypotheses or return relative/partial coordinates. The evidence contract should require auditable observed provenance for scale/orientation resolution, name whether units describe the original chart or a stretched copy, and preserve competing hypotheses when evidence cannot distinguish them. Full-paper averages must be supplemented by tiny-object retention and tail failures. These are necessary methodological decisions for the next version, not claims that the missing frontend and calibrated uncertainty machinery are already implemented.
