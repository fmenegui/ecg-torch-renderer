# M3 independent QA review

Date: 2026-10-23. Reviewer: `m3_independent_qa`. Baseline: PR21 `f4f6caef3b59b5830e93284cccadbc6855fc701e`.

**Decision: the complete 160-result diagnostic comparison passes independent numerical, provenance and postprocessing audit. One truncated fallback PNG was recovered from a verified native duplicate, with the original failure preserved. These results support the stated retrospective comparison. They do not establish universal, clinical or commercial superiority.**

Execution lock: `a667f1cbbe3dc3334df1721a821af063691240eb0b3fdd7b72af5150f1127b3c`. Protocol: `83debefdb2b1bd33735c8883c0f0d3d17acc590a309ba82e79ac55a785013be0`. Primary assessment is 32 already-inspected synthetic captures from eight patients, five executable methods and 160 case/method outputs. 1 development capture with five method outputs is feasibility only. The complete 24-capture development pool was not scored and is not aggregated.

## What the controls establish

| Evidence | Independent result | Scope |
|---|---:|---|
| Restored data and query audit | 1,939 checks pass | 112 capture hashes, 224 query domains, checkpoint identity and 11 reconstructed post-noise opaque masks. Zero queries in those opaque interiors. |
| Mathematical metric oracle | 50 checks pass | Exact, abstaining, NaN, scale, reflection and patient-weight controls. A standalone oracle alone would be insufficient. |
| Actual metric core | 20 tests pass | Fixed denominators, accepted NaNs counted wrong, one shared origin offset, D4 isolation, explicit footprint, patient grouping and paired case retention. |
| Kaggle measurement field | 10 checks pass | Actual field source, exact corners in four rotations and 500 analytic projective/bilinear interior queries. No neural accuracy inferred. |
| Actual execution wrapper | 10 tests pass | Original preprocessing order, authentic full/partial map semantics, RGB separation, model/inference failures, timeout status, output budget and resume integrity. |
| Actual postprocessor | 6 tests pass | RGB success requires actual resampling, confidence abstention is separate, D4 stays diagnostic, common support is counted and failed supported-layout rows remain. |
| Independent result auditor | 3 tests pass | Rejects a fabricated reported score and a dropped scheduled case. Uses synthetic panels only so far. |
| Frozen execution bindings | 194 checks pass | 89 source/config files, five weight/reference assets, 56 query archives, patient/backend/view balance, CPU threads and 16M output budget. |
| Matched interpolation primitives | 763 checks pass | 10 scenes, three arms, all interior/boundary metrics and shared-edge seams independently recomputed. |
| Sampling repair candidate | 32,558 checks pass | 280 scenes, seven arms, independent polygon-area reference and full metric audit. No native benchmark or production change. |
| Actual RGB sampler | 44 checks reproduce thin-stroke loss | Exact affine geometry. At 4 px/mm, 0.1 mm strokes disappear at two of five phases. Thick 1 mm controls remain at full contrast. |
| New composition development probe | 831 recomputation checks pass | All 12 scenes and five arms retained, fixed analytical truth and metrics recomputed from saved arrays. |

The actual npj geometry wrapper was also reviewed against its original source functions, native scale estimator and half-pixel resize/perspective chain. The source owner's parity controls compare the original geometry functions. Generic and GeorgeMoody2024 remain separately named presets.

## Problems found before diagnostic freeze

The initial runner omitted the pilot grid points required by periodic banding removal. This would have failed execution and differed from frozen M2 preprocessing. It was restored to the original detector, input-conditioned banding removal and optional redetection. Independent controls exercise both branches against the actual runner source.

The Kaggle forward-measurement wrapper initially rejected exact native-frame pixel centres at boundaries. Inclusive bounds were corrected before freeze. Its original neural stages, author bilinear field, original pixel-centre convention and RGB sampler remain in the authentic arm. The correction changes measurement availability at boundaries, not the author rectification algorithm. The source owner additionally demonstrated a successful actual-model run on an author-provided normalized image. That is a wiring control, not an unbiased performance score.

Model-load errors now leave scheduled outputs in the denominator. Inference timeouts are explicit. Resuming checks the result's execution lock, case identity and prediction archive hash. The case deadline is cooperative SIGALRM. It does not prove hard preemption inside a long native C/C++ operation, so observed elapsed overruns must be reported.

## A reproduced cause of apparent deletion

The frozen RGB sampler can completely miss a thin stroke even with an exact affine map and no confidence mask applied to RGB. The control uses a 201 by 201 source image at 10 pixels/mm and a vertical black stroke outside the trusted grid region. A one-source-pixel stroke is 0.1 mm wide. At 4 output pixels/mm, source-column phases 11 and 14 disappear completely. Phases 10, 12 and 13 retain contrast. At 8 pixels/mm all five retain some contrast. Ten-source-pixel, 1 mm controls retain full black contrast at all five phases and both densities.

This is the actual frozen `render_complete` implementation with `cv2.remap` and `INTER_LINEAR`, without an antialias prefilter or area integration. It is a sampling failure, not an error in the exact supplied geometry. The analytical result does not estimate its frequency in the image bank and does not establish 8 pixels/mm as generally sufficient.

The saved contact sheet `sampling-stroke-contact-sheet.png` shows original, 4 px/mm and 8 px/mm samples in the same physical viewport. Nine original/output PNGs are preserved with hashes. The display uses nearest sampling of the actual stored values, without added contrast or an invented stroke. The methodological repair must evaluate output footprint integration, anisotropy and physically justified density against exact area truth. Increasing density alone is not a demonstrated general solution. A separately frozen sampling-candidate study has now completed and does not change M3.

## Sampling repair candidate, independently audited

A separate component probe retains 280 strip scenes, two exact affine maps including anisotropy/shear, seven orientations, ten phases, thin and thick widths, and all seven sampling arms. At the same 4 px/mm density, midpoint integration of the original-source reconstruction removes the four complete thin-stroke disappearances seen in 140 native thin-stroke cases. The coarse 2 by 2 pixel-cell reconstruction still loses two cases, so finite supersampling is not a guarantee.

For 0.1 mm strokes, the native mean absolute relative dark-area error is 16.9329%. The 8 by 8 bilinear-integration candidate reduces it to 0.15936%. However, mean pixel MAE is lower with 2 by 2 bilinear integration than with 4 by 4 or 8 by 8, so greater quadrature order is not a monotonic improvement across objectives. The integration order and reconstruction model must be chosen against a declared preservation objective and a measured budget.

The independent audit passes 32,558 checks. It uses polygon clipping rather than the candidate's CDF to check 4,276 physical output-pixel areas and 4,200 source-pixel observations. All 1,960 saved metrics and 14 summaries were recomputed. All 280 native outputs and 120 selected candidate quadratures were reproduced from direct affine coordinates. At half-LSB source-quantization ties, CDF and polygon arithmetic differ by one level in 78 checked instances, with no larger disagreement. These instances remain disclosed.

Map preparation, cache memory and RGB application cost remain separate. Coordinates were cached across synthetic images sharing the same exact map, and the host was shared with other work. There is no end-to-end speedup claim, no production default change and no proof for nonlinear photos, clinical waveforms, arbitrary contrast or a universal density. This is a concrete sampling prototype with validated analytical benefits and retained failures.

## Methodological decisions

Primary scoring removes only one global translation, estimated from 64 pre-frozen disjoint paper anchors and reused in both domains. It never repairs scale, rotation, reflection, shear or a local warp. The D4 table is a separate diagnostic of orientation versus distortion. Neither table certifies automatic absolute time or voltage baseline.

Every scheduled case remains, including an unavailable registration. Native partial coordinates and actual RGB rectification are separate outcomes. A preserved original has zero operational RGB rectification coverage even when local coordinates are useful. Unrestricted finite-field errors remove confidence selection inside an explicit native footprint, with nonfinite/fold/hole counts disclosed. This is necessary to investigate apparent region deletion but does not itself measure waveform stroke survival.

Correct coverage, wrong coverage and total coverage must be shown together. A low conditional risk obtained by rejecting most of the paper is not evidence of a better rectifier. Common-support errors supplement the full-domain results and must show how many queries remain. Accepted nonfinite coordinates count as wrong in M3, unlike the older M2 convention. M3 origin-gauge percentages cannot be compared directly with old M2 percentages.

The new shared projective/residual prototype has a coherent C0 construction. It removes projective interpolation bias on planar cases. The saved probe also retains regressions under high localization noise, an oblique crease and coherent half-sheet error. An independent SVD-based edge calculation also reproduces substantial seams for separate per-cell homographies on oblique curved and creased controls. The shared residual construction avoids that particular C0 defect. Its homography acceptance is a numerical guard, not a calibrated physical-accuracy certificate. It has not modified frozen M2, has no native RGB delivery evaluation and cannot yet be promoted as generally better.

## Limits that stay in the final report

The panel reuses data already inspected during M2. It is a paired retrospective diagnostic comparison, not an untouched assessment or a clinical validation. Renderer-exported dense truth and independent analytical controls serve different roles. Analytical positives do not eliminate renderer or training-domain bias.

Kaggle means the pinned third-place implementation with its original author weights, declared CPU float32 adaptation and separately recorded layout assumptions. It does not mean every Kaggle solution. Preserving its known sampler convention does not make that convention ideal. The unknown weight-license metadata must not be converted into a redistribution permission.

Dotter/Gridder have not been executed. Official API documentation exists, but no authorized credential was supplied and the exposed response schema does not provide a dense pixel-to-mm field. A patent-inspired geometric primitive cannot be labelled a commercial-model score. Commercial rectification performance remains unavailable, not failed.

Timing is descriptive on a shared host with models already loaded. Successful cold-start elapsed and per-method peak RAM were not measured, so this is not a cold-start or isolated-memory benchmark. Not every upstream Python/NumPy RNG entry point was seeded, which limits a claim of exact repeated inference, even with pinned code and weights. No post-freeze replay or tuning is justified merely to improve a diagnostic score.

No common waveform-extractor comparison, independent real-image geometric gold or source-annotated ECG stroke-retention assessment has been completed. The new stroke studies are analytical component controls. Therefore geometry scores, exact original archiving and a numerically valid inverse must not be called complete ECG preservation or clinical superiority.

## Complete diagnostic audit

The independent saved-result audit passes **50,223 checks**, covering exactly **160 of 160 scheduled case/method results**, eight patients, 32 captures and both render backends. Every source, weight, input, query and prediction hash matches. Every archived score was replayed. Primary translation and G/W/C/R were recomputed independently. No case was dropped or replaced. There are no observed case-wall or standalone-native-pipeline overruns of the declared 180-second window.

The independent summary audit passes another **821 checks**, reproducing the source-result hashes, 1,280 track/domain CSV rows, patient/backend means and all 20,000 paired patient-bootstrap draws for every metric, track and domain. Kaggle's R is undefined in all 20,000 draws because registered C is zero. It must not be reported as zero risk.

| Method | Primary paper G | W | C | R | RGB resampled | Operational RGB-field G |
|---|---:|---:|---:|---:|---:|---:|
| our-B0 | 55.020% | 15.952% | 70.972% | 22.48% | 18/32 | 46.438% |
| our-M2 | 55.272% | 15.665% | 70.937% | 22.08% | 18/32 | 46.931% |
| npj-generic | 10.667% | 77.025% | 87.692% | 87.84% | 32/32 | 10.667% |
| npj-george-moody-2024 | 10.631% | 77.061% | 87.692% | 87.88% | 32/32 | 10.631% |
| kaggle-third | 0.000% | 0.000% | 0.000% | undefined | 3/32 | 0.000% |

These fractions use the predeclared fixed physical queries and equal patient/backend weighting. Primary orientation remains native. The D4 diagnostic raises M2 paper G from 55.272% to 68.166%, demonstrating a substantial orientation contribution without changing primary ranking rules. The small M2 versus B0 primary G difference is 0.252 percentage points. Its descriptive paired 95% interval is 0.000 to 0.690 percentage points, and the panel was previously inspected. This is not a confirmatory superiority test.

Our arms resample only 18 of 32 captures. Fourteen preserve the original and expose partial coordinates. Their operational full-field C is therefore 56.25%, rather than their roughly 71% trusted-coordinate C. Both npj presets return 32 rasters, but their large physical errors remain in W. Kaggle returns three rasters and fails on 29. All three returned fields have 2,508 zero grid nodes and 2,408 invalid inverse cells. Its predeclared compatible 3x4+r1 stratum contains only one case, one patient and one backend. There is no adequate supported-task assessment of Kaggle here.

There are 103 actual resampled images and 28 selected original-preserved images. All 131 selected images decode fully after the single disclosed recovery. Pixel decode validity does not certify resampled waveform fidelity.

## Preserved artifact incident and recovery

The first full audit stopped at `211723-2/our-B0/original-preserved.png`. It was 589,961 bytes and ended during the ninth IDAT chunk, without the remaining data or IEND. Its SHA256 was `da634ca2aebd211cdd6738a1a74228900eac1f93e26bdd4f8439050a01292b01`. The pre-repair full-image scan found this one failure among 131 images. Cause is unknown.

The same native invocation had also saved `original-rgb.png` and `output-original-rgb.png`. Those files were byte-identical, complete and pixel-for-pixel equal to the hash-verified bank RGB. Under explicit root authorization, the bad bytes and original stat were preserved in `artifact-incidents`, and only the corrupt export was restored atomically from the valid native duplicate. The recovered image SHA256 is `a5d0b664bd7ee75d244d0da8f21ad84c2e65948254d0c7a45220d42f3ca6f9d6`.

The recovery receipt proves that lock, result JSON and prediction hashes remained unchanged. No inference, geometry, score, status or timing was changed. The affected case was already a non-resampled operational abstention, so the numerical comparison is unchanged. The original truncated export is not retroactively claimed valid. Future image output should use atomic writes and full pixel-decode validation, rather than accepting PNG dimensions alone. Frozen production code was not changed by this recovery.

## Execution hardware and remaining validity limits

The CPU model string is AMD EPYC 9V74 80-Core Processor. The container sees nine logical/affinity CPUs, with an eight-CPU cgroup quota and 8 GiB memory. The native launch used Torch/OpenMP two threads, OpenCV one thread and OpenBLAS one thread from the first invocation. These are loaded-model timings on a shared host. No isolated peak-RAM, cold-start or GPU-performance conclusion follows.

The central repair priorities remain grid identity/scale ambiguity, native orientation, reliable full-map acceptance and source-footprint sampling. The new area-integration and shared-residual prototypes have analytical evidence, retained failures and explicit costs. Neither has been promoted as a new production default by this diagnostic study. Commercial Dotter/Gridder execution, representative compatible-layout evaluation, independent real-image geometry and waveform/stroke fidelity remain separate evidence requirements.
