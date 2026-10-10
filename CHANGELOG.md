# Changelog

## 0.4.0

- Add `ECGSupervisionLayer` and `SupervisionConfig` for aligned RGB, lead masks,
  union/BW/WB views, per-lead boxes and per-instance masks/boxes.
- Support reordered/subset channels, soft/binary coverage and visibility.
- Skip baseline generation and legacy instance-box work in the advanced path.
- Export aligned NPZ targets, pixel-edge JSON boxes and lossless BW/WB PNGs.
- Add a complete small U-Net example and supervision integration tests.
- Include supervision configuration in detailed profiling reports.
- Replace library docs with an English guide, local search, generated API
  reference, responsive themes and 125 actual generated figure recipes.
- Extend the English Colab notebook with U-Net and supervision profiling.

## 0.3.0

- Add optional compiled raster kernels and batched deterministic coverage.
- Preserve extrema and physical layout semantics against frozen 0.2.1.
- Add operator/kernel/memory profiling with separate warmed timing.
- Package a standalone library, console tools and GPU Colab notebook.

## 0.2.1

Frozen numerical/performance reference, retained as the immutable archive
`references/ecg-gpu-generator-v021.zip`.
