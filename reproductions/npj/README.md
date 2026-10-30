# npj Digital Medicine reproduction

Primary sources:
[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12891466/),
[original code](https://github.com/Ahus-AIM/Open-ECG-Digitizer/tree/97a15087d4abcda843da8c58ee74b1d8f47e6f9a),
[development data](https://huggingface.co/datasets/Ahus-AIM/Open-ECG-Digitizer-Development-Dataset).

The native path retains segmentation, perspective correction/cropping,
grid-size estimation, lead-text/layout identification and signal extraction.
Experimental nonlinear dewarping is disabled in the original provided presets.
The native output is in **uV**, not mV. No reference signal reaches inference.

## Protocol discrepancy to reproduce explicitly

| Setting | Article Table 4 | Original public `unet.yml` |
| --- | ---: | ---: |
| Epochs | 45 | 100 |
| Weight decay | 0.001 | 0.01 |
| Cosine scheduler batches | 20000 | 10000 |

`paper_table4` overrides these three settings and retains the original model,
loss, optimizer classes and augmentation definitions. `public_config` retains
the public training settings. `lead_text_public` uses the separate original
lead-name network configuration. We do not choose a profile after seeing test
scores or silently describe the public YAML as the paper's exact training run.

The audited training harness explicitly initializes GradScaler and serializes
both inner AdamMuon optimizer states. Its completed-epoch resume is tested on
an independent tiny interrupted-versus-uninterrupted optimizer control.
CUDA training/resume parity and original-run stochastic parity remain untested.

## Original dataset

The release contains 4112 training and 217 validation rows, each with original
and T0 image/mask pairs. Thus the 4329 rows represent 8658 image/mask pairs.
The dataset card advertises 75,032,904,415 download bytes. This release was not
fully downloaded in the CPU preparation phase.
The 72 Parquet shards are pinned to dataset revision
`c9665befe4d79eb0295602c568ddae73bcce1014`, with official LFS SHA256 hashes
and byte lengths in `development-data.lock.json`. The download and extraction
commands verify these identities. Counts alone do not establish original data.

`prepare_data.py` extracts both pairs without resizing or re-splitting. It
retains byte hashes and rejects repeated source row IDs across the supplied
shards. Row IDs alone do not establish biological patient separation.

```bash
python -m reproductions.npj.fetch_data --dest /workspace/npj-parquets

python -m reproductions.npj.prepare_data --parquet-root /workspace/npj-parquets \
  --out /workspace/npj-data --require-complete

python -m reproductions.npj.train --upstream /workspace/sources/npj \
  --profile paper_table4 --train-dir /workspace/npj-data/train \
  --validation-dir /workspace/npj-data/val --data-receipt /workspace/npj-data/data-receipt.json \
  --out /workspace/npj-main-paper

python -m reproductions.npj.train --upstream /workspace/sources/npj \
  --profile public_config --train-dir /workspace/npj-data/train \
  --validation-dir /workspace/npj-data/val --data-receipt /workspace/npj-data/data-receipt.json \
  --out /workspace/npj-main-public

python -m reproductions.npj.train --upstream /workspace/sources/npj \
  --profile lead_text_public --out /workspace/npj-lead-names
```

`--smoke --device cpu` deliberately reduces training to two image patches and
one validation patch. It tests the execution path and is not a training result
for either article cohort.

## Native inference and scoring

```bash
python -m reproductions.npj.infer --upstream /workspace/sources/npj \
  --preset ahus --image /workspace/ahus/scan.png --out /workspace/npj-ahus

# Original author entrypoints, executed from the pinned upstream directory:
python -m src.digitize --config src/config/inference_wrapper_ahus_testset.yml \
  DATA.images_path=/workspace/ahus DATA.output_path=/workspace/npj-native
python -m src.evaluate --config src/config/evaluate.yml \
  digitized_dir=/workspace/npj-native ground_truth_dir=/workspace/ahus \
  results_csv=/workspace/npj-native-metrics.csv
```

Preserve the original Ahus filename/layout hint policy when replaying that
cohort. `infer.py --layout-hint` makes a supplied hint explicit. Use the separate
George Moody preset and original 2024 scoring protocol for the Emory results.
A generic layout catalogue on PM-ECG-ID is a transfer experiment.
`infer.py --segmentation-checkpoint` and `--lead-checkpoint` select retrained
weights explicitly and retain their hashes, separately from published weights.

The source audit executes the native scorer unchanged. It retains two concrete
controls: exact equality produces undefined SNR, and all-missing output retains
the initial reported missing fraction of zero. The native main loop also finds
existing output files rather than enumerating every scheduled input. Our audit
track exposes absent outputs and perfect matches separately. These controls do
not establish why the authors obtained any particular cohort score.

## Comparison with the paper

| Original Ahus stratum | Reported SNR | Reported RMSE | Reported PCC | Reproduced here |
| --- | ---: | ---: | ---: | --- |
| Scanner | 19.65 dB | 19.3 uV | 0.987 | Pending original cohort and GPU execution |
| OnePlus | 12.19 dB | 42.8 uV | 0.947 | Pending |
| iPhone | 10.47 dB | 51.3 uV | 0.935 | Pending |

The original Ahus cohort contains 266 ECGs and 1596 device/page images. Although
the npj article offered access upon reasonable request, the subsequent
[ECG-Image-Database release](https://www.kaggle.com/datasets/physionet/ecg-image-database)
now provides the Ahus component publicly as D2. Its
[data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13402943/) describes 266
records and 1596 images. The full 5-second trace of all 12 leads is printed
across two six-lead pages. Device and page mappings must come from the released
metadata and must preserve the original layout-hint policy.

The distinct 35,595-image component combines D0 PTB-XL and D1 Emory. Selecting
only D1 would not recreate that denominator. The native 2024 scoring protocol
and the exact original subset/variant crosswalk still need reconciliation.
Capture-condition targets remain in `../reported-results.json`.

Version 2 metadata was retrieved and a resumable file-inventory acquisition was
started. Full image/reference acquisition and original-cohort scoring have not
completed. Data availability is therefore not the current access blocker. See
`../fetch_catalog.py` and `../evidence/public-clinical-data-release.json`.

The local native fixture run returns a `[12,5000]` array in uV. No reference or
printed-lead mask accompanied that smoke test, so its finite fraction is not a
digitization-accuracy or printed-support-coverage result.
