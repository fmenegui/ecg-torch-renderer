# Kaggle 2025–2026 top-five reproduction

Competition:
[PhysioNet: Digitization of ECG Images](https://www.kaggle.com/competitions/physionet-ecg-image-digitization).
This is distinct from the 2024 George Moody challenge used by the npj Emory
evaluation. They do not share an interchangeable leaderboard score.

## Candidate selection

All five official writeups were freshly retrieved. Metadata and content hashes
are in [`selection.json`](selection.json). The selected candidate is
[Takashi Someya's second-place solution](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/2nd-place-solution).

| Rank | Reproduction complexity found in the public material |
| --- | --- |
| 1 | Ten waveform models, several prediction strategies. Complete training release not identified in the consulted writeup. |
| 2 | Original final notebook, public training refactor, six waveform models in final A and useful single-model results. Selected. |
| 3 | Original final notebook v15 has fifteen waveform checkpoints. GitHub default training is a single configuration. |
| 4 | Separate annotation, grid/index models, FCMAE pretraining and multiple five-fold waveform experiments. |
| 5 | Orientation model, five keypoint folds, two main waveform models, a tall-range refinement model and pseudo-label registration. |

Selection is an engineering judgment about public availability and complexity.
External robustness and cost rankings remain to be measured. The chosen simple
ablation is **Whole B7**, with reported public/private scores **22.93/22.65 dB**.
Final submission A uses **six** waveform models and reports **23.37/23.27 dB**.
A B7 run must never be compared as though it reproduced the final ensemble.

## Native inference

`replay.py` executes the original submission notebook v6. The external model
dataset is pinned to version 2 and shared Heng assets to version 3. Source and
weight identities are checked. Changes are limited to mounted paths, serial
GPU-0 execution, retaining intermediate outputs and headless diagnostic plots.
All six original waveform models remain enabled.
The replay also refuses missing or unexpected checkpoint keys, instead of
silently accepting a partially loaded network.

Use a separate environment with `../requirements-kaggle.txt`. The retained first
checkpoint audit documents a model-name parsing incompatibility with current
timm 1.0.30. This is an environment failure, not a digitization score.

```bash
python -m reproductions.kaggle.fetch_assets --dest /workspace/kaggle-assets
python -m reproductions.kaggle.replay \
  --notebook /workspace/kaggle-assets/notebook.api.json \
  --assets /workspace/kaggle-assets \
  --assets-lock /workspace/kaggle-assets/assets.lock.json \
  --competition-data /workspace/competition \
  --out /workspace/kaggle-native-six
```

The input interface is the original `test.csv` plus `test/{id}.png` layout.
Public sampling/lead lengths may reach inference. Reference waveforms may not.
All scheduled image identities are saved. A failed native image's submission
zeros are retained for official scoring and marked as fallbacks in the audit.
Native four-row arrays are preserved before conversion to submission format.

## Retraining

The pinned public repository is a later refactor whose README warns of possible
regressions. Its default encoder is B0, not the selected B7. The source audit
also finds different waveform decoders between its `predict.py` and the original
notebook on an analytical bimodal probability map. This finite difference is
not proof of any particular leaderboard impact.

Use `train.py` for the author's original training entrypoint with explicit
architecture profiles, data paths and fold. A data contract supplies absolute
`csv_path`, `mask_dir`, `ecg_dir`, `csv_sha256`, `num_synthesis_data` and
`split_provenance`. Original preprocessed data and fold identities must be
verified, and native-versus-refactor inference parity must pass before calling
retrained weights equivalent to a final-submission model.

```bash
python -m reproductions.kaggle.train --upstream /workspace/sources/kaggle_second \
  --profile whole_b7 --data-contract /workspace/kaggle-data-contract.json \
  --fold 0 --out /workspace/kaggle-whole-b7
```

The six profiles expose the architectures in final A. Public default
hyperparameters do not prove the exact settings of all six original training
runs. Synthetic-data quantities and missing per-model details remain explicit
unknowns. The wrapper will not guess a synthesis quantity.

## Comparing scores

Run the original official 2025 metric on the original submission and solution
rows. Preserve its alignment, ratio cap, image aggregation and final dB
conversion. It is not the mean of per-lead dB values. Scoring our fixed-denominator
audit on another cohort is a separate result.

The original public and private leaderboard cohorts have not been scored in
this execution. No native six-model GPU forward pass or full retraining has
completed. All comparison targets remain pending in `../reported-results.json`.
