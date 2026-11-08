# M2: guarded localization refinement and new-patient assessment

Noisy grid positions can distort an otherwise valid cell map. This experiment adds an optional, bounded refinement of detected positions, with fixed lattice IDs and a conservative curvature guard. Original detections and RGB remain available. The CLI default is still B0, `--refinement-strength 0`.

The frozen strength 1.0 improves correct geometric coverage modestly on 64 new synthetic captures from eight patients. It reduces wrong coverage, but leaves the severe error tail essentially unchanged. This is an optional research result, not evidence of universal or clinical superiority.

## Paired results

The grid domain is primary. Correct means physical coordinate error at most 0.5 mm. G and W use all fixed queries as denominator. C = G + W and R = W/C. Means weight each patient and each rendering backend equally, rather than pooling all image pixels.

| Cohort | Method | G correct | W wrong | C returned | R wrong/returned |
|---|---|---:|---:|---:|---:|
| Development, 48 captures / 6 patients | B0 | 72.2059% | 1.5652% | 73.7712% | 2.1217% |
| Development | Soft 1.0 | 73.4955% | 1.2075% | 74.7030% | 1.6164% |
| New-patient assessment, 64 captures / 8 patients | B0 | 79.3523% | 2.8577% | 82.2100% | 3.4761% |
| New-patient assessment | Soft 1.0 | 79.7592% | 2.4264% | 82.1856% | 2.9523% |

Assessment differences in percentage points, with descriptive 95% paired patient bootstrap intervals:

* G: +0.4069 [+0.0567, +0.9740].
* W: -0.4313 [-1.0091, -0.0659].
* C: -0.0244 [-0.0483, -0.0056].
* R: -0.5238 [-1.1984, -0.0761].

Bootstrap uses 20,000 draws, seed 1973, resampling whole patients with both backends, and recomputes aggregate R. These conditional intervals are descriptive, with eight independent assessment groups and no multiplicity adjustment. Absolute performance on this cohort must not be compared with an older cohort as if it were a paired improvement.

The gain is concentrated in 3D LLVM: G 64.3997% to 65.2069%. Torch G changes from 94.3049% to 94.3115%, with a delta interval including zero. Paper-domain G changes from 70.6047% to 70.9251%, and W from 2.9925% to 2.6372%.

## Protocol and leakage controls

`prepare_nextgen_study.py` freezes 48 development captures using the six previously inspected development patients and adds 64 captures from eight PTB-XL 1.0.3 patients absent from all 1,280 previous patient groups and 77 exclusions. Deterministic selection was fixed before opening the new images. All 112 captures rendered successfully. Each patient contributes six torch and two true 3D LLVM captures.

The existing checkpoint is unchanged: SHA256 `8115e99c3517dc0b99974b9e64744aab9eb961fd0da06f1e84a65a1e45c3dbf8`. There is no new neural training. The candidate strengths are 0, 0.05, 0.15, 0.5 and 1.0. Development maximizes G subject to frozen overall and per-backend W/R constraints. Strength 1.0 was selected before scoring any new assessment image.

Every paired arm uses the same detector output, inferred IDs, B0 evaluation gauge and physical queries. Gold is used for metrics and evaluation-only signed axes/permutation/integer-origin alignment. No gold scale, shear or affine fit is used to rescue predictions. Absolute automatic orientation is not evaluated. Image, checkpoint, source, cache and query hashes are recorded. The development summary replay matches the earlier B0 result.

The gold-node export audit checks composition of the renderer's own maps, not independent physical truth. New patients provide patient separation, not a new generator or external camera domain. No competitor implementation was run in this experiment.

## Method and failure behavior

`robust_refinement.py` fits a robust normalized projective prior, then uses sparse adjacent-ID curvature constraints on residuals. IRLS weights and ranked detector scores are bounded, not calibrated probabilities or covariances. Missing IDs are not bridged. Shifts are capped and backtracked to preserve valid observed quad orientations. This does not certify global injectivity or fix wrong IDs and major/minor pitch ambiguity.

The coherent-curvature guard abstains when the smoothing prior conflicts with repeated spatial curvature. It can preserve a real fold or leave a coherent detector error untouched. Weak and noisy folds can escape. CLI candidate rejection returns to B0 using input-only checks. Original detector positions, candidate positions and effective positions are archived separately. `geometry_accuracy_verified` remains false.

On assessment, ten images are effectively refined, 53 trigger the guard and 54 preserve original coordinates exactly. There are 42 full mesh proposals in both arms, 21 cases with partial coordinates, and one without eligible coordinates. These are coordinate results before RGB sampling, not a count of successfully resampled images. All 22 construction-fallback flags occur when the refiner already left points unchanged. There are zero fallbacks after an actual refinement attempt. Seven refined cases reach the IRLS iteration limit, and three converge. Median geometry time is 0.187 to 0.338 seconds, excluding network inference and RGB sampling on a shared CPU.

## Negative controls and remaining errors

* Ten analytical scenes and nine arms give 90 retained results. At strength 0.15, three noisy-projective seeds improve RMSE from 0.5277 to 0.2757 mm, 0.5966 to 0.3351 mm, and 0.3412 to 0.1343 mm.
* Smoothing an exact clean crease without the guard creates 10% wrong coordinates at strength 0.15 and 22.67% at strength 1. The guard preserves B0 exactly on this control, rotated/relocated strong creases and an oblique crease.
* The guard loses a useful improvement on a noisy crease, where unguarded RMSE would improve from 0.2798 to 0.1787 mm. A weak crease escapes with about 0.003 mm residual bias.
* Development case `01025-6` increases W by 2.7344 grid-domain percentage points and 2.3926 paper-domain points. It remains explicitly reported.
* Assessment has no regression above the predeclared severe diagnostic thresholds. Smaller regressions remain: `208114-1` loses 0.0488 grid G points and increases paper W by 0.1953 points. `220288-1` loses 0.3906 paper G points.
* Assessment common-support RMSE barely changes, 2.15814 to 2.15690 mm. Severe topology/identity errors remain. For example, `206852-2` retains RMSE 14.89 mm, p95 35.20 mm and C 3.49%. Case `215864-6` retains RMSE 9.82 mm, p95 20.07 mm and C 64.94%. The next method must address those rather than merely increase smoothing strength.

The separate native-tile detector pilot uses eight development captures from one patient. Micro F1 changes from 98.3017% to 98.3083%, adding 28 true and 26 false positives. Median CPU latency increases from 7.80 to 10.10 seconds. Four cases use native tiles and four hit the budget fallback. This configuration is not promoted and was not used in the primary paired assessment.

## Reproduction and evidence

The two bank archives reconstitute `build/nextgen-bank`. The measurements archive contains per-case queries, eligibility, predictions, detector caches, analytical arrays, original result JSON, the frozen selection, unchanged detector checkpoint and prior cohort metadata. Do not regenerate the original upstream raster fixtures.

```bash
python examples/prepare_nextgen_study.py --sources build/study-inputs/previous-cohort-sources.json --root build/nextgen-bank
python examples/evaluate_nextgen_refinement.py --bank build/nextgen-bank --checkpoint build/study-inputs/joint-1973.pt --out build/reproduced-m2 --split validation
python examples/evaluate_nextgen_refinement.py --bank build/nextgen-bank --checkpoint build/study-inputs/joint-1973.pt --out build/reproduced-m2 --split test --selection build/reproduced-m2/selection.json
python examples/summarize_nextgen_study.py --root build/reproduced-m2 --out build/reproduced-summary --include-test
python examples/refinement_control_study.py --help
python examples/rectify_denoising_grid.py --help
python examples/build_nextgen_report.py --out build/m2-report.html
```

`summary.json` contains all grouped metrics, descriptive intervals, coverage and failure counts. `cases.csv` retains all five development strengths, the two frozen assessment arms and both query domains. `analytic-summary.csv` retains all analytical arms. Protocols, provenance, execution versions and SHA256 receipts are adjacent.

Fifty focused tests passed. The two GitHub workflows, Library contracts and Digitization reference, passed on implementation commit `0eb6a3f1a2b52f61737b24ec795048359102cb76`. Local full-suite execution ended without a summary. The remaining modules were rerun in separate processes. Only two checks were blocked by absent upstream binary artifacts, `tests/fixtures/raster-v021.npz` and `references/ecg-gpu-generator-v021.zip`. Complete-checkout CI preserves and checks those assets. Sixteen capture scenarios were generated and visually inspected. Four post-hoc illustration cases were also generated at 4 px/mm. Their eight CLI maps and eligibility masks match the evaluator exactly in both query domains, with maximum coordinate difference 0 mm. Six outputs were resampled and two B0 validation outputs preserved the original RGB after mesh rejection. All eight archived originals match their inputs exactly. See `validation-status.json` and the final QA review for the exact scope.
