# Read-only audit of saved npj M3 outputs

Date: 2026-10-23. Scope: six diagnostic capture keys and two already-frozen official presets, twelve case-method pairs. This audit performs no neural inference, changes no frozen file and makes no quality-based method choice. It does not change any published score or threshold.

## Fidelity result

`audit_saved_fidelity.py` and `saved-fidelity-audit.json` reproduce and check the saved artifacts. All twelve pairs pass:

* Source archive SHA256 matches the scored input.
* Original, unchanged Cropper reconstructs exactly the recorded destination points from the saved source corners.
* Half-pixel resize, original perspective coefficients, portrait rotation and recorded row crop reconstruct the saved source-to-output-pixel and source-to-mm matrices with **maximum difference 0**.
* Replaying the original Cropper sampler on the original min-max-normalized RGB, followed by recorded orientation/crop and the benchmark's uint8 rounding, reproduces every decoded native PNG pixel **exactly**.
* Recomputed native physical predictions for all 64 anchors and both 4,096-query domains have **maximum difference 0 mm** from the saved predictions. All eligibility masks agree exactly.

This replay does not require the neural probabilities. It verifies how their saved geometry was consumed. Earlier controls independently verified the continuous map against actual bilinear sampling of coordinate ramps and exact parity with original upstream wrapper geometry methods on a portrait analytical chart.

The adapter remains SHA256 `fd4a022d273c7682d14f99c3ff4910dbb8c7417d2f50e61b8fe2f13ac59d4aa9`. The audited benchmark lock is SHA256 `a667f1cbbe3dc3334df1721a821af063691240eb0b3fdd7b72af5150f1127b3c`.

## A zero-coverage case is native crop degeneration

Case `220288-0`, layout `1x1-long`, is not an empty returned PNG or an adapter-coordinate exception. Both Generic and George Moody produce the same source quadrilateral. Its average top edge is near y=415.51 and bottom edge near y=397.53. Its signed vertical extent is approximately **−17.98 pixels**, yielding a negative approximate aspect ratio.

The unchanged original Cropper therefore computes destination x coordinates from approximately **−15,638 to +16,662**, far outside the 1,024-pixel output canvas. The native perspective map magnifies a very narrow portion of the ECG by about 33 times. Visual inspection of the saved PNG confirms a highly enlarged segment of a trace and grid, rather than the whole document.

Only one of the 4,096 grid queries falls inside that native output footprint, and no frozen anchor does. Thus the benchmark correctly records native grid coverage 0.0244% and operational registered-map coverage zero. It must not be described as an exact zero native footprint, a thrown inference error or a calibrated rejection of the image.

A separate analytical call to the original `Cropper._get_indices` shows its bound-degeneration mechanism: when all mass lies in bin 20, the 0.02/0.98 quantiles return bounds **(20, 19)**. This proves a possible upstream crop inversion. The exact cached segmentation probabilities were not retained, so it is a plausible mechanism for this single-row case rather than proof that its probability distribution was exactly that control.

## A positive case confirms useful authentic behavior

Case `208777-0`, layout `6x2+r1`, produces sensible corners and destination bounds. It retains 55 eligible anchors. Both presets have grid G=C=97.65625%, with RMSE about **0.24948 mm** after the permitted translation-only origin registration. D4 diagnostics do not change its result. The native image visibly retains the multiple trace rows and physical grid, with the original occlusions still present.

This confirms that the benchmark is not categorically rejecting the external method or assigning it a wrong unit convention. It also demonstrates why whole-document conclusions must not be inferred from one failed format.

## High errors reflect native scale and orientation limitations in these cases

| Case | Generic grid RMSE | GM2024 grid RMSE | Diagnostic interpretation |
|---|---:|---:|---|
| `220288-2` | 20.98 mm | 8.22 mm | Same projective map, different inferred x/y spacing. Native scale matters materially. Neither D4 diagnostic fixes it. |
| `208777-2` | 63.95 mm | 73.23 mm | Generic anchor spans are about 185×132 mm versus reference 357×259 mm, roughly half size. D4 diagnostic leaves the same result. |
| `220288-6` | 190.41 mm | 177.69 mm | The image carries a recorded 180° post-rotation. D4 diagnostics reduce RMSE to 63.21 and 75.93 mm respectively, leaving substantial non-orientation error. |
| `208777-6` | 132.63 mm | 132.73 mm | The image carries a recorded 270° post-rotation. D4 diagnostics reduce RMSE to 63.73 and 67.14 mm respectively, again leaving substantial scale/global-shape error. |

The original presets expect 30 to 70 major lines and infer scale using a periodic major/minor template. Some audited synthetic formats exceed that assumed grid extent. For example `208777-2` has visible major-grid column IDs 13 to 91, spanning 78 intervals. Period aliases under these fixed bounds are a plausible contributor to the roughly half-sized native field. The saved scale estimates and predicted spans substantiate a scale discrepancy, but no frozen neural probability map or candidate autocorrelation score curve was saved. Therefore this audit cannot uniquely separate a wrong harmonic from segmentation failure or global projective-model inadequacy.

The two prescribed presets must remain separate. These examples include a case improved by the GM weighting and another worsened by it. Choosing the favorable preset using scored queries would be an oracle change. No such change is proposed or executed.

## Methodological consequences

1. Reject inverted or nearly degenerate crop geometry using input-only checks before displaying a magnified patch as a rectification. Preserve the original separately. This is a future method change, not a repair applied to this benchmark's npj arm.
2. Physical scale estimation needs explicit ambiguity accounting and a wider-format strategy. A hard line-count range is a substantive format prior, even when the neural segmentation and global rectifier are layout-independent.
3. Keep native orientation accuracy separate from a D4 diagnostic. A diagnostic orientation rescue does not certify scale or local shape, and is not used in primary ranking.
4. Report native RGB footprint separately from operational registration availability. This is especially consequential for tiny footprints with no valid anchors.
5. Attribute the tested configurations precisely: authentic npj geometric stage with Generic and GM2024 presets, published source and actual U-Net checkpoint. Downstream layout recognition and waveform extraction are not evaluated by these scores.

No artifact or frozen inference code was modified by this audit. Input-only replay images were compared in memory and not substituted for the native outputs.
