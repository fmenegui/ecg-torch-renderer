# Optimization plan after 0.12.0

Status: proposal. Nothing here is implemented or measured yet. Every expected
gain below is a hypothesis to test, not a result. The acceptance gates in
`AGENTS.md` apply to every phase.

## Where the library stands

| Area | Evidence | Gap |
| --- | --- | --- |
| GPU speed | Only measurement: 0.4.0 on a Tesla T4, about 31 images/s at 1024x768, batch 4, compiled. Eager full forward issued about 10.6k kernels | The 0.9 to 0.12 paper, material and point models were never timed on a GPU |
| CPU speed | 0.5.0: about 3.5 images/s at the full preset | Too slow to train without a GPU |
| Raster | Brute force: every instance evaluates a capsule distance at every pixel and subsample | About 47 M distance evaluations for about 20 k ink pixels per 1024x768 page (roughly 2400x) |
| Stroke geometry | `research/stroke-width/probe.py` | Perpendicular width falls from 3.00 px to 1.27 px at slope 10. QRS strokes print about one third as thick as baselines |
| Dense targets | float32 masks: 144 MiB lead masks and 180 MiB instance masks at B=4, 1024x768 | Binary or bounded targets are stored at full float precision |
| Domain gap | Grid pilot: online-trained F1 0.916 on torch images, 0.025 on ray-traced images | The fast training path does not cover the 3D distribution |
| API | `ECGRenderer.forward` has 18 keyword arguments. `_render` spans about 370 lines | Hard to extend, test in isolation or compile as a whole |

## Phase 0: measure before changing anything

1. Run `notebooks/colab_gpu_profile.ipynb` and `ecg-renderer-profile` on the
   current release at 512x384 and 1024x768, batch 1, 4 and 16, eager and
   compiled. Record median, p90, peak memory and kernel count.
2. Add `torch.profiler.record_function` ranges per stage: parameter sampling,
   instances, raster, grid and text, paper material, geometry inverse, lighting
   and shadows, camera, JPEG, supervision. The profiler then reports the share
   of each stage directly.
3. Publish the result as `PERFORMANCE_<GPU>_V0120.json` and replace the
   historical T4 numbers in the README.

Exit criterion: a stage-level time and memory breakdown on a named GPU. The
order of Phases 2 and 3 follows that breakdown.

## Phase 1: isotropic strokes, with the legacy raster preserved

The current raster is frozen bit-for-bit against v0.2.1 by
`test_raster_matches_frozen_v021_all_30_layouts`. That fixture stays as is.

1. Add `RenderConfig(stroke="legacy" | "isotropic")`, default `legacy`.
2. Isotropic mode: each pixel takes the minimum capsule distance over the
   segments of its own column and of the `ceil(thickness)` neighbouring
   columns on each side. This is a small fixed window, so it stays a dense
   tensor operation without host synchronization.
3. Tests against an independent analytic reference: straight lines at slopes
   0 to 50 and a circle must keep perpendicular width within 5 % of
   `line_width`. The extremum test must still pass. Legacy mode must still
   match the frozen fixture with `atol=0`.
4. Rerun the grid benchmark and a lead-mask U-Net with both modes. Switch the
   default only if the measured transfer to ray-traced images does not get
   worse, and record that decision in the CHANGELOG.

## Phase 2: a sparse raster kernel

Replace dense gather-style coverage with scatter-style rasterization.

1. Each subsample segment becomes a capsule with a bounding box of a few
   pixels. A Triton kernel (CUDA) writes its coverage with an atomic max into
   the ink, lead and instance targets. A dense torch fallback stays for CPU,
   and the current dense kernel remains as the reference implementation.
2. The work becomes proportional to the stroke area rather than to
   `instances x samples x H x W`. The extremum envelope fits the same scheme as
   short vertical capsules.
3. The kernel naturally produces isotropic strokes. The legacy mode keeps the
   current code path.
4. Gates: max absolute difference against the dense isotropic reference below
   `1e-5` on all 30 layouts. Identical results across batch splits. No host
   reads. Throughput measured on the Phase 0 GPU.

Risks: atomics make floating order nondeterministic, so use max (order-free)
and not sums. Triton becomes an optional dependency on CUDA only.

## Phase 3: fewer kernels and less memory in the scene

The 0.4.0 profile shows about 10.6k kernels per eager forward. At this scale
launch overhead and memory traffic dominate, not arithmetic.

1. Split `_render` into pure stage functions with typed inputs and outputs,
   for example `layout`, `raster`, `document`, `geometry`, `lighting`,
   `camera`, `encode`. This is a prerequisite for compiling stages separately.
2. Compile each stage with `torch.compile(fullgraph=True)` and keep the random
   sampling eager as today. Then try CUDA Graph capture of the whole forward
   for fixed shapes, with the generator state as a graph input.
3. Evaluate the geometry inverse and shadows at reduced resolution and
   upsample, where the stage breakdown shows they dominate. Validate with the
   existing dense-solver reference in `paper_audit.py`.
4. Measure kernel count, latency and peak memory before and after each step.

## Phase 4: compact targets

1. `SupervisionConfig(target_dtype=torch.float16 | torch.uint8)`. Binary
   targets as `uint8`, soft targets as `float16` with documented error.
2. Optional packed instance masks: 15 instances fit in one `int16` bitfield per
   pixel, unpacked in the loss.
3. Losses in `training.py` accept the compact types and convert per tile.

Expected effect: lead and instance masks drop from 324 MiB to 81 MiB in
`uint8` at B=4, 1024x768, before packing.

## Phase 5: narrowing the online-offline gap

1. Mix precomputed offline geometry and visibility banks into the online path
   (the extension point already exists) and measure transfer at equal budgets.
2. Repeat the grid pilot with several training seeds and more test patients,
   since the current confidence intervals cover only four test patients.
3. Run the same protocol on lead masks, not only on grid points.

## Phase 6: API and repository

1. Group `forward` outputs into an `Outputs` dataclass instead of 18 booleans,
   keeping the old keywords for one minor release with a deprecation warning.
2. Rename `models_v05` and `schema_v05` with compatibility aliases.
3. Publish the documentation from CI to GitHub Pages and stop committing the
   generated `docs/` site. It is stale today: it still reports 0.8.0 and lacks
   the realism, grid-point and architecture pages.
4. Move large research assets (`.npz`, galleries) to release assets or Git LFS.
   Rewriting history to shrink `.git` is destructive and needs a separate
   decision by the owner.

## Order and dependencies

Phase 0 comes first. Phases 1 and 4 are independent of the measurements and
can start in parallel. Phase 2 builds on the isotropic reference from Phase 1.
Phase 3 needs both the Phase 0 breakdown and the stage split. Phase 5 needs
Phase 1 to separate the stroke effect from other causes of the gap.
