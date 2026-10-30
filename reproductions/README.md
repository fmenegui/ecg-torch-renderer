# ECG digitization: independent reproductions

Started 2026-10-30 on an independent branch from `main` at
`6a4e0cbc011234e31592a9ba79f45e09187389b2`.
Earlier M2–M7 experiments are prior observations, not acceptance criteria.
This work does not promote the existing rectifier or change production defaults.

## Current status

| Work | Implemented and locally executed | Still needed |
| --- | --- | --- |
| npj Digital Medicine | Original source and both published weight files verified. Complete native chain executed on one original test fixture. Both original networks trained in explicitly reduced CPU and Runpod L4 GPU engineering checks. Full native inference executed on an author fixture and public development images. A new 24-photo, eight-patient GPU cohort is saved separately. Full-data training has a safely verified epoch-13 checkpoint out of 45 main epochs. Native evaluator controls executed. | Original 45-epoch and public-config training, lead-text training at full scale, original Ahus and Emory cohort scoring. |
| Powerful Medical | Independent public-description Gridder, declared Dotter hypothesis, training/inference entrypoints, analytical geometry controls. | Original annotated Dotter bank, exact architecture and checkpoint, full normalization and waveform validation. This is not the commercial implementation. |
| Kaggle | Top-five review, second-place selection, original six-model notebook replay adapter, six training profiles, source audit against the later refactor. All eight preprocessing/waveform checkpoints strictly loaded. Native six-model notebook executed on a Runpod L4 and produced a waveform for one public development image. A separate A40 replay completed all eight checkpoint loads but all six eligible external photos failed native stage-zero normalization. Fallback zeros are not recovered signals. Earlier failures are retained. | Full retraining, native original-cohort scoring, refactor/checkpoint parity. A single B7 model is an ablation, not the six-model final submission. |

**No headline result has been reproduced yet.** Every target in
[`reported-results.json`](reported-results.json) is labeled author-reported.
Every execution has its own receipt under [`evidence/`](evidence/).
Control tests, training completion and matched-cohort reproduction are separate
claims. The real GPU receipts and the retained first replay failure are under
`evidence/runpod-gpu-pilot-2026-10-30/` and
`evidence/runpod-native-replay-2026-10-30/`. Full article scores remain unverified.
See [the GPU review](evidence/GPU_REVIEW.md) for the measured pilot and limits.
The later [prospective GPU study](../research/hybrid-digitizer-20261030/prospective/README.md)
preserves the primary bank, separately labeled continuations and training progress.

## Scientific rules

1. Pin original source, checkpoints, data bytes, inference settings and scorer.
2. Run each native method before changing it. Preserve native failures and
   intermediate arrays. Run corrected audits in a separate track.
3. Match cohort, paper layout, printed time windows, units, sampling rate,
   alignment, missing-data treatment and aggregation before comparing numbers.
4. Retain every scheduled image and lead. Numeric output coverage is not
   physical visibility, morphological fidelity or clinical acceptability.
5. Treat source-image variants as grouped observations. Biological patient
   independence needs source evidence, not an invented patient ID.
6. Never replace missing trace data with an allegedly recovered flat signal.
   A zero used by an official scorer or submission fallback is labeled as such.
7. Keep training data, development data and a fresh external test separate.
   A failed external case becomes development after inspecting or tuning to it.
8. Do not compare a normalization component with a complete digitizer's SNR.

The common audit metric in [`metrics.py`](metrics.py) retains the original printed
denominator, reports finite returned samples and adds a support-constrained
alignment diagnostic. It is deliberately separate from the native npj and
official Kaggle scores. [`compare.py`](compare.py) refuses direct headline
comparisons when identities or protocols differ.

## Environment and source recovery

Python 3.12 is required by the native npj project. Our completed local checks
used CPU Torch 2.7.0, torchvision 0.22.0 and the package versions in
[`requirements-common.txt`](requirements-common.txt). These dependencies are a
recorded reproduction environment, not a claim that every package matches the
authors' historical environment.

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r reproductions/requirements-common.txt
python -m reproductions.bootstrap --dest build/reproduction-sources
ECG_REPRO_NPJ_UPSTREAM="$PWD/build/reproduction-sources/npj" python -m unittest discover -s reproductions/tests -v
```

For GPU workers install the corresponding Torch 2.7.0 CUDA 12.8 wheels first.
Source hashes and provenance are in [`sources.lock.json`](sources.lock.json).
Original third-party code and weights are fetched outside the tracked source.
Their original attribution and licences remain attached to those upstreams.

The subsequent public ECG-Image-Database version 2 provides the Ahus clinical
component and the combined PTB-XL/Emory component. It is a data acquisition
route, not evidence that our original-cohort scoring has completed. The pinned
metadata and cohort-reconciliation questions are in `sources.lock.json` and
`evidence/public-clinical-data-release.json`. A resumable metadata-only inventory
can be obtained without downloading the 36.4 GB of file bytes:

```bash
python -m reproductions.fetch_catalog --out build/ecg-image-database-catalog
```

Keep dataset version 2 fixed. Verify the imaging-variant workbook, original
image identities, device/page labels and scorer mapping before declaring the
public release an exact replay of either article cohort. The combined
35,595-image component includes both D0 PTB-XL and D1 Emory, not D1 alone.
The completed public file inventory is retained in
`evidence/public-clinical-file-inventory.json`. Its 22,806 files sum exactly to
the advertised byte count. D2 contains the expected 1,596 raster images.
D0/D1 exposes 13,770 raster images and 2,954 PDFs, so it cannot yet be equated
with the article's 35,595 images. PDF contents and release/subset composition
need reconciliation. File enumeration alone does not explain that difference.

Read [`npj/README.md`](npj/README.md),
[`powerful_medical/README.md`](powerful_medical/README.md) and
[`kaggle/README.md`](kaggle/README.md) for exact entrypoints and open questions.

## Runpod execution

The requested GPU lane is Runpod. A local missing credential or unavailable
CUDA is recorded as a blocker. GPU commands refuse a silent CPU substitution.
The small CPU runs under `evidence/` are engineering checks only.

The repository Actions secret is `RUNPOD_API`. It authenticates successfully.
The secret is read inside Actions and never printed or copied to tracked code.
`Runpod ECG execution` reads live stock, prices and the official image catalog,
then runs a bounded 30-minute engineering pilot on one GPU with a temporary
30 GB durable volume. An independent pod deadline guard and the Actions
controller both enforce shutdown. Each owned pod is externally checked absent.
Results are hash-verified and uploaded before its temporary volume is deleted.
The current CLI 3.0.1 does not expose the older advertised terminate flags.
REST v2 returned null runtime telemetry while GraphQL exposed working SSH,
so the controller uses the corroborated GraphQL port mapping explicitly.

The separately bounded full-training workflow uses the complete 72-shard,
75 GB original release, a 220 GB volume and at most five five-hour segments.
It retains batch 12, 1024-pixel patches and the 45-epoch `paper_table4` profile,
then the 30-epoch `lead_text_public` profile. Every resumed invocation reloads
a completed-epoch checkpoint. An unfinished epoch is discarded and restarted.
The public source's historical font bank and stochastic author run are not
available, so font hashes and these limitations remain in the receipts.
The original batch failed the 24 GB L4 capacity control, so full training
requires at least 48 GB. The maximum GPU price is USD 0.99/hour and the total
GPU envelope is USD 24.75. A new full-shape control runs before data acquisition.
Storage and provider rounding are additional. After all segments, an unfinished
training volume is retained for recovery and is still billable. Completed
training volumes are deleted only after checkpoint artifact upload.
No source-profile completion is treated as a matched-cohort result.

Provision only after a fresh stock, template, region and price read. Use one
GPU at a time. Save data, configuration, logs and checkpoints on a network
volume. Run a bounded pilot before committing to complete training.
[`runpod_job.py`](runpod_job.py) is an inner process-group timeout and stop
guard. It does not provision infrastructure or prove that billing ended.
The external controller must independently verify pod shutdown and preserve
outputs before deleting resources created for these experiments.

```bash
# Inside the newly created Runpod pod, with /workspace on the durable volume:
python -m reproductions.runpod_job --out /workspace/jobs/npj-main --timeout-seconds 3600 -- \
  python -m reproductions.npj.train --upstream /workspace/sources/npj \
  --profile paper_table4 --train-dir /workspace/npj-data/train \
  --validation-dir /workspace/npj-data/val --data-receipt /workspace/npj-data/data-receipt.json \
  --out /workspace/npj-main
```

The timeout above bounds an initial attempt. It does not redefine full training
or authorize claiming that a partially trained model reproduced the paper.

## Comparison gates

| Gate | Required evidence |
| --- | --- |
| Native implementation | Original source and strict checkpoint identity, real forward pass, saved output. |
| Retraining | Original data/split identity or explicit adaptation, full effective config, complete training log and checkpoint. |
| Reported result | Exact original cohort and native scoring protocol, with all outcomes retained. |
| Common benchmark | One frozen cohort, independent printed windows, common denominators, patient grouping and uncertainty. |
| Public library candidate | Reproducible improvements in signal fidelity and coverage, adverse cases, resource measurements and clear unsupported cases. |

The final goal is a public library with measured benefits. These reproductions
establish baselines and expose assumptions. They do not yet establish superiority
or clinical benefit.
