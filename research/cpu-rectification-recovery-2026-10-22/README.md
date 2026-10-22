# CPU rectification recovery and partial-support audit

The larger study had already completed. This continuation recovers that state,
finishes patient-cluster uncertainty estimates and repeats a small, frozen
image-only comparison without starting a GPU or retraining a model.

## Changes and observed behavior

The Kaggle geometry-only constructor used to require waveform and metric
artifacts that it never invokes. Verification and explicit downloading now
honor `rectification_only=True`. The seven Heng geometry artifacts still pass
the same size and SHA-256 checks. The complete digitizer keeps its full checks.

The npj image adapter used to reject the upstream identity fallback because
an empty maxima array had shape `(0,)` instead of `(0, 2)`. Empty diagnostic
points now preserve the returned RGB. A warning emitted by the pinned upstream
dewarper explicitly records its reported identity fallback. This warning capture
depends on the frozen upstream message and is not a generic TPS success test.
No upstream source, checkpoint or algorithm was modified.

The local rectifier now reports supported-cell coverage, four-connected cell
components and largest-component coverage. Its denominator is the bounding
rectangle of observed relative integer identities. It is not the complete page,
gold grid-ID agreement or waveform coverage. No missing cell is filled.

## Frozen exploratory comparison

Inputs are exactly the first eight recovered gallery views, `01152-0` through
`01152-7`, from one public PTB-XL patient. These are JPEG exports of synthetic
test captures, not the original lossless cached bank. All adapters receive the
same bytes. Input hashes are in `images.json`. The checkpoint is `joint-1973`,
selected previously using validation only, SHA-256
`8115e99c3517dc0b99974b9e64744aab9eb961fd0da06f1e84a65a1e45c3dbf8`.
Inference uses CPU float32 and four torch threads. Our display output uses
2 pixels/mm with the previously declared 5 mm major-grid assumption.

| Method | RGB returns | Interpretation |
| --- | --- | --- |
| Local detector and lattice | 8/8 | All explicitly partial, with substantial holes in some views |
| Kaggle geometry stages | 0/8 | All fail stage0 document-reference localization |
| npj RGB adapter | 8/8 | Two report TPS failure and use upstream identity fallback |

For our output, `01152-2` has 18.13% observed-extent cell coverage and seven
connected regions. `01152-7` has 32.23% coverage and 110 regions. A returned
partial image therefore cannot be counted as successful complete rectification.
The six other npj outputs were produced before fallback instrumentation was
added. Their original execution log contains no reported TPS failure. The two
empty-point cases were rerun after the adapter correction, with both attempts
retained in `references-attempt-before-fix.json` and `references.json`.

This comparison has no common full-page gold coordinates and no waveform
extraction. It establishes no geometric superiority, clinical validity or
real-photograph generalization. Eight views of one patient are not eight
independent patients. Image-return counts differ from accuracy.

## Larger held-out synthetic results

`patient-bootstrap.json` applies the existing, frozen summarizer to all four
1,024-image test reports. Each paired bootstrap resamples 128 patients 2,000
times. Seeds are reported separately and are not treated as patients.

| Seed | Backend | Joint minus point F1 | Patient-bootstrap 95% interval |
| --- | --- | --- | --- |
| 1973 | torch | +0.00991 | [+0.00824, +0.01177] |
| 1973 | 3D | +0.04405 | [+0.03259, +0.05609] |
| 20261021 | torch | -0.00031 | [-0.00261, +0.00222] |
| 20261021 | 3D | +0.00875 | [+0.00105, +0.01739] |

These intervals are conditional on the trained models. Both 3D trials improve
point F1, but the effect size varies by seed. The torch gain does not repeat
in the second seed. This is not uncertainty over future initializations.

`geometric-support.json` separately summarizes the existing gold-node geometry
measurements, including zero-support failures in the denominator. For
`joint-1973`, median true-node support is 92.30% on torch captures and 75.85%
on 3D captures. Fifty of 256 3D images have zero supported true nodes. These
are original-bank measurements and must not be mixed with JPEG rerun counts.

## Reproduction

With the locked JPEG input list and separate upstream assets:

```bash
python examples/audit_rectification_support.py \
  --images inputs/images.json --checkpoint selected.pt \
  --source /path/to/pinned/Open-ECG-Digitizer --assets /path/to/locked/assets \
  --out build/cpu-audit
python examples/summarize_diverse_study.py --root /path/to/recovered/results
```

For an explicit geometry-only asset acquisition, call
`fetch_assets(path, rectification_only=True)` outside inference. The full
digitization CLI keeps its existing default acquisition scope.

## Validation and infrastructure

23 final focused tests pass. The complete integration run executes 378 tests:
351 pass, 25 skip and two error because the original files
`tests/fixtures/raster-v021.npz` and
`references/ecg-gpu-generator-v021.zip` are unavailable in this recovered
checkout. Neither expected fixture was regenerated. The complete checkout's
CI must verify both original regression assets before merge. The adapter's
two formerly failing npj cases also complete with the instrumented fallback.
A 16-scenario multi-layout gallery was exported and visually inspected on CPU.

The live pod read returned `rl2ouqsgfgedwe` as `EXITED`. The live volume read
returned `wxt8cq1hh1`, 50 GB, CA-MTL-1. No infrastructure mutation or new
training was performed in this continuation. The user photograph was not used
in this comparison or uploaded.

## Next measurable step

1. Evaluate image-only page localization and higher-resolution inference for
   small papers, with selection performed on validation patients only.
2. Improve relative lattice identity assignment while tracking connectivity,
   orientation, coverage and error together. Preserve explicit missing support.
3. Recover the bank's gold coordinate fields and compare all methods on common
   support, retaining failures and full-page coverage in the denominator.
4. Evaluate waveform preservation and independently annotated real photographs.
