# Runpod measurements of the GPU renderer, 2026-11-17

Branch `research/renderer-gpu-20261117`. Artificial signals for profiling;
public PTB-XL and ECG-Image-Database D0 for sim-to-real. Every pod was deleted
and its absence verified by the controller. GPU spend ≈ USD 2.52.

## 1. GPU profile (run 44093688811)

NVIDIA RTX 2000 Ada (16 GB), Torch 2.7.0+cu128, USD 0.24/h, ~13 min of pod.
Raw report: `renderer-gpu-profile` artifact, `renderer-profile/report.json`.

| Preset | B | Eager img/s | Compiled img/s | Peak extra MiB |
| --- | ---: | ---: | ---: | ---: |
| fast (384×256) | 16 | 151.7 | 197.3 | 498 |
| balanced (512×384) | 4 | 70.0 | 83.6 | 250 |
| balanced (512×384) | 16 | 59.9 | 92.4 | 996 |
| full (1024×768) | 4 | 12.2 | 23.2 | 996 |
| full (1024×768) | 16 | 11.4 | 20.6 | 3984 |
| full + all 15 opt-in flags | 4 | 3.0 | — | 1046 |

- CUDA kernels per forward: ~4,350 (balanced/full), 9,632 with all flags.
  Balanced B=4 keeps the GPU busy 23 % of the wall time; B=16 reaches 75 %.
- Cold compilation 14–27 s; compiled warm calls are 1.2–2× faster.
- CPU/GPU parity with fixed parameters (clean, direct camera, no JPEG): image
  max |Δ| = 1/255, targets ≤ 1e-7 (labels 9e-6). The same seed with sampled
  parameters gives unrelated captures (different generators).
- TF32 on vs off: 0.18 % of pixels differ by more than one 8-bit code.
- A failing `torch._assert_async` (blur=5) raises "device-side assert
  triggered" and leaves the CUDA context unusable for the process.
- The CUDA unit tests did not run: `tests/` is not in the transferred sources.

## 2. Sim-to-real pilots

Same deterministic split (seed 1973): 600/60 PTB-XL training/validation
records per arm; 25 D0 test patients × 4 capture kinds = 100 real images.
Open-ECG-Digitizer UNet trained from scratch, 1500 steps, batch 4, crop 1024.

Published npj weights (run 44096808365, RTX 4000 Ada), mean SNR dB
[patient bootstrap 95 %]: all 3.50 [2.89, 4.17]; digital 12.31; color scan
0.44; gray scan 0.77; phone photo 0.48. 100/100 returned.

KID (Inception pool features, 200 rendered training images vs the 100 real
images; lower is closer; std over subsets ≈ 0.005 overall):

| Arm | All | Digital | Color scan | Gray scan | Phone |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy (run 44096808365) | 0.146 | 0.208 | 0.199 | 0.302 | 0.150 |
| full (run 44114365619) | 0.134 | 0.212 | 0.187 | 0.260 | 0.142 |
| full-v2 (run 44114365619) | 0.143 | 0.231 | 0.202 | 0.261 | 0.141 |

`full-v2` = `full` + `prefilter_warp=True` + a 64-entry Mitsuba geometry bank
with baked ambient occlusion (`examples/build_geometry_bank.py`). Its sampled
captures differ from `full` beyond the two changes, so per-kind differences of
~0.01–0.02 are within composition noise. No KID improvement is shown.

**Retrained-arm SNR is unavailable.** Run 44096808365 timed out during legacy
inference (13 s/image serially). Run 44114365619 ran six inference processes
on one 24 GB GPU: 83 of 100 images failed with CUDA OOM in each arm, so its
SNR rows are void. The fix (serial inference, saved checkpoints) is in
d2b0484; its run 44137343645 found no secure GPU in stock and allocated nothing.

## 3. Fine-tuning comparison (run 44195924404)

RTX A5000, USD 0.27/h, ~108 min. Each recipe fine-tunes the published
segmentation weights for 800 steps (lr 3.7e-4) on its own 600 pairs; the same
100 real D0 images, 40 s per-image limit (published weights never exceeded
24.7 s). Run 44191110758 failed before training (`_orig_mod.` keys) and is
excluded.

| Arm | All dB [patient 95 %] | Digital | Color scan | Gray scan | Phone | Returned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| published (no fine-tuning) | 3.50 [2.89, 4.17] | 12.31 | 0.44 | 0.77 | 0.48 | 100 |
| full fine-tuned | 0.47 [−0.51, 1.46] | 4.12 | −1.02 | −0.31 | −0.91 | 94 |
| full-v2 fine-tuned | 0.84 [−0.11, 1.87] | 5.16 | −1.07 | 0.05 | −0.77 | 92 |

Paired by patient (bootstrap 95 %):
- full-v2 − full: +0.37 dB [−0.42, +1.19]; gray scan +0.36 [−0.10, +0.94];
  digital +1.04 [−1.89, +3.98]. Not significant.
- full − published: −3.03 dB [−3.90, −2.18]; full-v2 − published: −2.66 dB
  [−3.51, −1.79]. Fine-tuning on either recipe's synthetic pairs makes the
  published digitizer worse on real images, most on digital images (−7 to −8 dB).

KID in this run: full 0.137, full-v2 0.149.

**Reading.** Neither recipe yet produces training data that helps a digitizer
already trained on ECG-Image-Kit-style data; at this scale both degrade it.
full-v2 trends better than full on downstream SNR but worse on KID, and
neither difference is established. Next measurements: larger test cohort and
multiple seeds, mixing synthetic with the original training data instead of
replacing it, and per-change ablations (prefilter alone, bank alone).

## Limitations

One seed, 25 patients and a short training budget. KID uses ImageNet features
and few real images per kind. No result here shows that any recipe transfers
better to real images.
