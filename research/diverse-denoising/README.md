# Diverse offline captures and joint point / restoration learning

This is an active experiment. Targets and training scripts are implemented.
No completed larger training result is claimed until the server outputs are
recovered and verified. Current execution status is recorded separately.

## Bank specification

The target cohort has 1,280 unique public PTB-XL 1.0.3 patients, with 1,024
training, 128 validation and 128 test patients. Folds 1–8, 9 and 10 determine
the split. Seventy-seven previous/public-photo-reference patients are excluded.
Each patient contributes one original 100 Hz, 12-lead, ten-second ECG in mV.
ADC conversion, channel order, sample count and WFDB checksums are verified.
No diagnoses or physiological waveforms are invented for the training bank.

Eight captures per source produce a target of 10,240 cached images at
1024x704. Six views use the torch document/capture renderer. Two use Mitsuba
CUDA path tracing at 64 samples per pixel. Every image and its supervision is
cached before training. There is no generation in the network's forward path.

The bank samples all 30 supported lead layouts, all 16 capture recipes and
six prescribed three-dimensional paper families: flat, curl, fold, multifold,
wrinkled and crumpled. It varies speed, gain, font, grid phase, grid contrast,
trace width, severity, lighting and camera position. Some physical captures
include a missing corner, edge damage, hole or occluding object. Views also
sample quarter rotations with an explicitly tracked pixel-center transform
and letterboxing. This is a parameterized stress bank, not every possible
camera or paper observation and not a proof of photographic realism.

Cached optical corruptions include sensor noise, defocus and motion blur,
JPEG with loss of resolution, unequal illumination, faded printing, color and
gamma variation, scanner banding, stains and opaque patches. The source capture
already contains scenario-specific materials and optical effects, so these
combine with the extra corruptions. Opaque patches remove loss/point support.
Impulse noise and a synthetic moire recipe are withheld from training and
model selection, and appear in an additional synthetic stress evaluation.
The moire recipe is an image stress pattern, not a calibrated monitor model.

## Paired supervision and architecture

The clean RGB target is rendered from the identical source signal, print
layout and physical calibration, with observation damage removed. It is
projected through the capture's identical inverse material field. Quarter
rotation and letterboxing transform RGB, clean target, visibility and point
coordinates together. Thus restoration targets live in the observed view,
while geometric rectification remains a separate problem.

`DenoisingGridNet` is a small U-Net-like model with a width of 24. Its inputs
are RGB, locally normalized RGB and original grayscale. It uses no absolute
page position, global grid IDs or lead-layout specification. Its heads predict
crossing heat, subpixel offsets, visibility and a residual RGB restoration.

Two variants share architecture, initialization, cached source images, crop
sequence, optimizer settings and step count. The point control has point and
visibility losses. The joint variant adds a masked color/edge restoration loss.
Both use seeds 1973 and 20261021, for four trials of 10,000 AdamW steps each.
Batch size is four, with deterministic 384x384 crops, BF16 forward, FP32 losses
and clipping at norm five. CPU DataLoader workers read the offline cache.
Point masks and offsets are rasterized on CUDA outside file-loading workers.
The restoration loss is 0.2 times color L1 plus 0.25 times edge L1 within that
term. Visibility loss has weight 0.1 and the point positive weight is 10.

Recovery checkpoints retain model, optimizer, RNG and sampling-index metadata.
The `--resume` CLI restores model, optimizer and RNG, continues the deterministic
crop index, and rejects mismatched bank/protocol signatures. Recovery files are
replaced atomically. Interrupted-versus-uninterrupted CUDA parity has not yet
been demonstrated, so exact numerical continuation is not claimed.

Validation backend-macro F1 selects thresholds from 0.3, 0.5, 0.7, 0.85 and
0.93. The best point detector and the best jointly supervised model are
identified separately from validation only. This rule is recorded before any
new test results. A point-only control's RGB head is unsupervised and must not
be presented as a trained denoiser.

## Geometry and evaluation

The native grid organizer remains available as a baseline. A new local filter
retains cells whose four edges obey its declared directional/spacing bounds.
This allows explicitly partial output after a global rejection. It does not
certify relative IDs or global injectivity. All images retain their native
status beside the local partial status, and rejected regions remain unsupported.

The final geometric transform always samples the original RGB. The neural
restoration is a separate diagnostic output. No extracted ECG signal is
silently replaced by a restored trace. Calibration, lead identity, paper speed
and clinical interpretation are not inferred by the rectification CLI.

Testing includes crossing precision/recall/F1 at a two-pixel radius in the
1024x704 observation, relative grid-ID agreement, geometric error after an
evaluation-only similarity alignment and coverage of visible true grid nodes.
All failures remain in the image denominator. Conditional geometric residuals
are shown beside support. Restoration reports pixel error on visible support,
which does not establish clinical waveform fidelity. Paired bootstrap intervals
resample patients and are conditional on the trained model. Seeds are reported
separately, not treated as independent patients.

Public Kaggle/npj reference assets are fetched directly from their publishers
on the server after training, with the existing source/weight locks. Their
image adapters invoke no waveform extraction. Acquisition errors are recorded
separately from algorithm errors. A reference image return is not a quantitative
geometric win. The current job prepares a fixed 32-image qualitative subset.

The user's uploaded ECG photograph is development material after the previous
inspection. It is excluded from cloud upload and training. New clinical-photo
gold annotations remain necessary for an external accuracy claim.

## Infrastructure and data minimization

The larger initial staging bundle was rejected by automatic review because
it might disclose private ECG data or third-party assets. No resource was
created by that rejected attempt. The first replacement upload was 316,405 bytes of source code and public
dependency declarations. It passed GPU/setup tests but stopped before training
when the URL for an exclusion list returned HTTP 404. Its GPU was removed and
logs recovered. The corrected 316,985-byte package also carries 77 pseudonymous
public PTB-XL exclusion identifiers, each verified against the official public
metadata. It carries no images, waveforms, clinical/demographic fields or
pretrained weights. The second GPU attempt failed on PhysioNet HTTPS timeouts before training and
was removed after log recovery. The new 318,367-byte code-only package fetches
the publisher public S3 source, retaining the full 100 Hz subset at
`/workspace/datasets/ptb-xl/1.0.3` on the persistent volume. It runs a 48-image,
two-model engineering smoke before the full study. Smoke patients overlap the
full cohort and do not constitute an independent benchmark.

The live catalog read selected one Secure Cloud A40 at USD 0.59/hour in
CA-MTL-1. A 50 GB high-performance volume stores bank/checkpoints. The current
owned pod is `rl2ouqsgfgedwe` and the volume is `wxt8cq1hh1`. Do not modify the
user's original stopped A4000 pod. A controller bounds compute duration and
removes its own GPU after recovery. The volume remains for reuse of the public dataset, cached bank and
checkpoints. It is not deleted by this training controller. Actual final resource state must be read back before reporting it.

Local validation: 11 focused tests pass. The full integration run executes
371 tests with 12 skips and two errors from the missing original fixture
`tests/fixtures/raster-v021.npz`. That fixture is not regenerated.

## Entry points

```bash
python examples/diverse_grid_study.py --cohort /path/to/cohort --root build/diverse --steps 10000
python examples/summarize_diverse_study.py --root build/diverse
python examples/rectify_denoising_grid.py --checkpoint selected.pt --image input.png --out build/rectified --device cpu
```

Public data source: https://physionet.org/content/ptb-xl/1.0.3/
License: Creative Commons Attribution 4.0.
Citation: Wagner et al., PTB-XL, doi:10.13026/kfzx-aw45.

Two additional patient-bootstrap tests pass after that integration run.
