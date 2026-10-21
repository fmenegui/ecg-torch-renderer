# PTB-XL 1.0.3, 100 Hz, unofficial mirror

Download the five ZIP parts and accompanying files from the repository's `ptbxl-1.0.3-100hz` Release:

https://github.com/fmenegui/ecg-torch-renderer/releases/tag/ptbxl-1.0.3-100hz

This repository is currently private. Repository access and GitHub authentication are required. This mirror does not make the repository public.

## Download

With GitHub CLI authenticated for this repository:

```bash
gh release download ptbxl-1.0.3-100hz --repo fmenegui/ecg-torch-renderer --dir ptbxl-100hz
cd ptbxl-100hz
sha256sum -c SHA256SUMS
python verify_ptbxl.py .
```

The verifier checks all original metadata and every waveform archive member without extraction. After successful verification, extract the five archives into the same directory, preserving `records100/` paths. Metadata `filename_lr` values refer to those paths.

## Contents

- All 21,799 original 12-lead records at 100 Hz, 10 seconds per record.
- Five ZIP parts with original WFDB `.dat` and `.hea` bytes.
- Original `ptbxl_database.csv`, `scp_statements.csv`, and `LICENSE.txt`.
- Per-file and per-archive SHA-256 inventory and verification script.
- No signal resampling, relabeling or physiological alteration.
- Does not include the 500 Hz subset or the complete upstream supplemental materials.

The release build verifies waveform dimensions, initial samples, channel checksums and a canonical SHA-256 inventory of all 43,601 source files against the independently acquired copy. It uploads a draft release, downloads and verifies every asset, then publishes it. The dataset is attached to the Release and is not added to Git history. The original patient groups and official stratification folds are retained in the metadata.

## Attribution and license

Wagner, P., Strodthoff, N., Bousseljot, R.-D., Samek, W., and Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45

Database paper: Wagner et al. (2020), PTB-XL, a large publicly available electrocardiography dataset. Scientific Data 7, 154. https://doi.org/10.1038/s41597-020-0495-6

Authoritative source: https://physionet.org/content/ptb-xl/1.0.3/

Public download source: `s3://physionet-open/ptb-xl/1.0.3/`.

License: Creative Commons Attribution 4.0 International. The original `LICENSE.txt` is included unchanged in the Release. Original authorship is preserved. Only the packaging and additional documentation/inventory were changed. This unofficial mirror is not affiliated with or endorsed by the authors or PhysioNet.
