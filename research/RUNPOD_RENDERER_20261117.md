# Runpod measurements of the GPU renderer, 2026-11-17

Branch `research/renderer-gpu-20261117`. Artificial signals for profiling;
public PTB-XL and ECG-Image-Database D0 for sim-to-real. Every pod was deleted
and its absence verified by the controller. GPU spend ≈ USD 3.37.

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

## 4. Why fine-tuning degraded the published digitizer (diagnosis, 2026-11-18)

Checked against the pinned Open-ECG-Digitizer code and samples of its public
development dataset (Ahus-AIM/Open-ECG-Digitizer-Development-Dataset, val
shard 0):

1. **Scale.** Native inference resamples every image so its longer side is
   3000 px (`resample_size: 3000`) before the UNet. The npj training images are
   ~2600–3200 px wide. Our pairs were 1600×1200, so fine-tuning taught the UNet
   grid and trace sizes at ~half the pixels per mm it sees at inference.
2. **Outside-paper label.** In npj masks the green channel (class 1,
   `text_background_class`) covers text *and everything outside the sheet*.
   `npj_mask` zeroes outside the sheet, i.e. class 3. Off-sheet area is 11–34 %
   of our training pixels.
3. **Trace label profile.** npj trace labels are soft (mean 0.21–0.24 on
   labelled pixels); ours are 0.47 and cover 2–3× the pixel fraction at our
   lower resolution.
4. **Replacement, not mixture.** Every arm trained only on our pairs; the
   published model's own data distribution was dropped.

These protocol mismatches can explain the −3 dB drop (−8 dB on digital images)
without any statement about photographic realism. The recipe comparison must
be repeated after fixing them.

## 5. Corrected protocol (runs 44279013642 and 44305448463, 2026-11-18)

Pairs rendered at 3000×2250 (npj inference scale), off-sheet pixels labelled
as the npj text/background class (`npj_mask(off_sheet='text')`), and the
published weights fine-tuned 800 steps on a **mixture**: 488 pairs from four
shards of the npj development set, plus our 600 pairs in the mixed arm.

Run 44279013642 (RTX 2000 Ada) rendered too slowly to finish inference and
`full-v2` training failed with a cuBLAS error while the KID Inception model
still held GPU memory; only its KID values are used. Run 44305448463 (RTX
A5000, USD 0.27/h, ~77 min) completed both arms on all 100 images:

| Arm | All dB [patient 95 %] | Digital | Color scan | Gray scan | Phone |
| --- | ---: | ---: | ---: | ---: | ---: |
| published | 3.50 [2.89, 4.17] | 12.31 | 0.44 | 0.77 | 0.48 |
| npj-only fine-tune | 3.27 [2.68, 3.88] | 11.41 | 0.42 | 0.62 | 0.61 |
| npj + full fine-tune | **3.84 [3.20, 4.53]** | 13.54 | 0.69 | 0.72 | 0.41 |

Paired by patient (bootstrap 95 %):
- npj+full − npj-only: **+0.57 dB [+0.30, +0.87]**, 19/25 patients better;
  digital +2.13 [+1.24, +2.98]; color scan +0.26 [−0.02, +0.72]; gray scan
  +0.10 [−0.02, +0.27]; phone −0.20 [−0.64, +0.11].
- npj+full − published: +0.34 dB [+0.12, +0.55].
- npj-only − published: −0.24 dB [−0.46, −0.04] (the fine-tuning itself).

KID against the same real images: npj development data 0.051 (digital 0.141,
color 0.099, gray 0.153, phone 0.077); our full 0.144–0.148; full-v2 0.152.
D0 printouts were produced with ECG-Image-Kit, so KID also rewards that
document style, not only capture realism.

**Reading.** Once the protocol matches the digitizer, adding our renderer's
pairs to the original data improves real-image SNR, almost entirely on
digital and color-scan images; photos show no gain. The full-v2 changes are
still unmeasured on SNR. One seed and 25 patients.

## Limitations

One seed, 25 patients and a short training budget. KID uses ImageNet features
and few real images per kind. No result here shows that any recipe transfers
better to real images.
