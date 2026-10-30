# Frozen hybrid engineering reproduction

This experiment combines the unchanged native Open-ECG-Digitizer, an optional
original-RGB grid correction, and the original Kaggle second-place six-model
submission. It is an engineering pilot, not an upstream-independent or
clinical validation. The public cohort has unknown overlap with the upstream
models' training data.

## Fixed inputs and comparison

`frozen-cohort.json` schedules 24 physical images from eight patients, paired
across scans, Samsung photographs and bent-paper photographs. Two patients per
layout cover 3x4, 3x4+1R, 6x2 and 12x1. Selection excludes our earlier grid
training/development patients, the M7 patients and the already inspected D0
patient. `selection-exclusions.json` retains the excluded source identities.
Publisher metadata, every photograph and every reference have sealed hashes.
The public release is DOI `10.5281/zenodo.13617673`.

`protocol.json` freezes the policy, 500 Hz, ten-second output axis, 10 mm/mV
assumption, 3000-pixel native preset and CUDA execution before new inference.
The policy was calibrated on known D0/M7 development outputs. In particular,
the 20 ms relative agreement window and 0.05 mV DC tolerance must not be
presented as independently selected or as clinical error bounds.

| Arm | Input and behavior |
| --- | --- |
| `npj_native` | Original RGB, original pinned networks, full native layout catalog |
| `grid_adaptive` | Our pinned grid observations, inferred full-frame mesh, bounded sampling of the original RGB, then the same native npj engine |
| `kaggle_native_eligible` | Original six-model notebook only for an inferred 3x4+1R profile, with public time metadata |
| `selective` | Accepted native specialist first, accepted geometry second, otherwise the exact direct result |

The selector rejects mismatched images, physical axes, layouts, fallbacks,
folded/incomplete meshes, excessive extrapolation, any lost direct sample and
shared-signal disagreement. It never fits output amplitude, shifts a returned
signal, blends samples, fabricates a missing lead or reads a reference.
Support checks are numerical rejection rules, not waveform correctness.

Every scheduled case remains in the full scoring denominator. Unsupported
Kaggle layouts and failed native series are saved as missing values. Notebook
submission zeros are not substituted for observed waveforms. References enter
scoring only after all outputs and decisions have been sealed.

Primary scores are patient-balanced printed coverage, raw NRMSE over every
printed sample with an explicit missing-sample penalty, and missing-lead counts.
Conditional aligned SNR and same-sample raw RMSE remain secondary. These are
our common comparison metrics, not the original paper or Kaggle leaderboard
denominators. `kaggle_native_eligible` is intentionally constrained by its native
profile and must not be advertised as a general all-layout baseline.

## Local replay and CLI

The source bootstrap and original asset acquisition are in `reproductions/`.
Use the pinned Torch 2.7.0/CUDA 12.8 stack in `requirements-common.txt`. The
native notebook runs in a separate process with timm 1.0.22. For geometry,
install Shapely 2.1.2 and the external Tesseract executable.

```bash
python -m reproductions.bootstrap --method npj --dest build/sources
python -m reproductions.kaggle.fetch_assets --dest build/kaggle-assets
python -m unittest discover -s tests -v
python -m unittest discover -s reproductions/tests -v

python -m ecg_gpu.digitization.hybrid_cli \
  --image ecg.png --npj-source build/sources/npj \
  --grid-checkpoint build/grid-checkpoint.pt \
  --device cuda --max-side 3000 --fs 500 --duration 10 --out build/hybrid-example
```

The CLI exposes the direct/geometry engine. The separately pinned six-model
notebook batch is integrated into `worker.py`. A generic specialist callback
is available in `SelectiveDigitizer`. The checkpoint default identity is
`8115e99c3517dc0b99974b9e64744aab9eb961fd0da06f1e84a65a1e45c3dbf8`.
Requesting another gain is currently rejected. Sample rate, duration and gain
are caller metadata and are not automatically inferred from arbitrary paper.
NaNs, canonical lead names, mV units, origin and the decision are saved.

Two audit replays reuse retained development measurements without training:

```bash
python examples/hybrid_retrospective_replay.py --measurements M7 --out build/m7-replay
python examples/hybrid_d0_replay.py --pilot-outputs GPU_PILOT4_OUTPUTS --out build/d0-replay
python examples/hybrid_sampling_study.py \
  --manifest M7_INPUTS/manifest.json --prior M7 \
  --upstream build/sources/npj --max-side 1500 --device cpu --out build/sampling-control
```

The 1500-pixel control is not the published preset. A recorded 3000-pixel CPU
attempt exceeded this engineering host's 8 GiB memory limit before producing
a prediction. The adapter now rejects that known resource combination and
does not silently lower resolution or substitute CPU for CUDA.

## Bounded Runpod execution

`.github/workflows/hybrid-runpod.yml` runs on a changed immutable `plan.json`.
Admission suppresses duplicate paid allocations for the same plan. Full local
contracts must pass before GPU provisioning. The global concurrency group is
dedicated to the hybrid pilot. The bounded pilot may run alongside the original
training. Before allocation, superseded pending hybrid workflows are cancelled
only if they have no jobs. An already running hybrid pilot instead defers the
new one. The original training workflow is never cancelled by this admission.

1. Run `41105577589` recovered the hash-verified selected checkpoint from the
   existing `ecg-diverse-offline-bank` volume in CA-MTL-1, mounted at `/archive`.
   Its owned recovery pod was deleted and the borrowed volume's presence was
   verified. The recovery was limited to 900 seconds and USD 0.60/hour, or USD
   0.15 GPU maximum. Generation 5 reuses that run's checkpoint artifact and
   verifies its exact hash before allocation, without another recovery pod.
2. A separate one-GPU pilot has at most 1800 seconds at USD 0.60/hour, or USD
   0.30 GPU maximum. Generation 5 uses 40 GB of container disk and no persistent
   or network volume. The prior network-volume creation was rejected by Runpod's
   USD 5 minimum-balance rule before any benchmark pod was allocated. Ephemeral
   mode is restricted to this short inference profile and cannot apply to full
   training, which keeps its durable checkpoints and network volume.

Stock and prices are read immediately before creation and created-pod prices
are checked again. Each owned pod has an independent deadline watchdog and
Actions cleanup with external absence verification. The maximum additional
GPU envelope is USD 0.45, plus temporary storage. The existing borrowed volume
retains its previous storage billing. No checkpoint recovery CPU was allocated
because the live catalog had no co-located CPU capacity.

The account credential stays in Actions and the watchdog's memory. It is not
part of source archives, experiment environments, receipts or checkpoints.
Only source-code directories are transferred. The worker fetches the pinned
public photographs itself and verifies the frozen manifest byte-for-byte.
The controller copies and hash-verifies results before deleting the inference
pod, then uploads them as Actions artifacts. The independent deadline remains
active if the controller fails. An abrupt pod failure before collection can
lose partial outputs on container disk. The frozen inputs and checkpoint remain
available for a faithful repeat. This is not a durable training configuration.

The grid branch now retains initial/final detections, confidence, relative IDs,
selected points, supported cells and complete meshes in `geometry-observations.npz`.
A rejection keeps the available observations and the last attempted stage.
The new `RectifiedNpjBackend` and `LandscapeRectifiedNpjBackend` are separately
named development controls. They do not replace the frozen pilot's native
downstream path or its selection policy. Their development runner is
`examples/hybrid_downstream_study.py`. The wider reference-assisted lag diagnosis
in `examples/audit_digitizer_time_axis.py` does not alter inference or scoring.
