# Completed larger synthetic study

The study generated 10,240 captures from 1,280 public PTB-XL patients and trained four models for 10,000 steps each. The test contains 1,024 images from 128 patients, split into 768 torch captures and 256 path-traced captures. Patient groups are disjoint across train, validation and test.

| Training | Mean test F1, torch | Mean test F1, 3D |
|---|---:|---:|
| Points only | 0.954281 | 0.843143 |
| Points plus restoration | 0.959079 | 0.869543 |

Means are across the two fixed seeds, not confidence intervals. Thresholds and the displayed model joint-1973 were selected on validation only. Point matching uses a 2-pixel radius at 1024x704. A high point F1 does not establish correct complete rectification. Qualitative examples include severe missing-support failures. No clinical or real-photograph accuracy is claimed. Reference inference execution does not establish superiority over Kaggle or npj.

All four recovered checkpoint hashes match the training records. Complete metrics, model weights and the first-32-example gallery were saved as user deliverables. The GPU was stopped with status EXITED verified after recovery. Its automatic cleanup had not occurred, and the persistent volume remains intact.
