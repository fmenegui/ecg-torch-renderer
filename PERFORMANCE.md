# Performance

The hot path uses batched Torch operations with no image-file I/O, NumPy rasterization or CPU text drawing. GPU-friendly structure alone is not a measured throughput result. Benchmark your actual GPU and training objective.

## Current paper model, CPU

The 0.5.0 paired benchmark uses Torch 2.14.1+cpu, two threads, ten timed calls and three warmup calls. Eager, compiled raster and compiled raster plus scene render the **same current model**. The optional historical backend is 0.4.0 eager, which renders a different paper distribution. Cold compilation is recorded separately and excluded from warm results.

| Preset | B | Eager ms | Raster ms | Raster + scene ms | Scene images/s |
| --- | --- | --- | --- | --- | --- |
| fast | 1 | 58.8 | 54.3 | 52.5 | 19.92 |
| fast | 4 | 155.4 | 140.7 | 136.2 | 29.61 |
| balanced | 1 | 79.1 | 78.4 | 63.1 | 14.72 |
| balanced | 4 | 425.0 | 264.5 | 242.3 | 16.34 |
| full | 1 | 460.8 | 324.0 | 315.2 | 3.21 |
| full | 4 | 1768.0 | 1158.1 | 1130.7 | 3.47 |

[Raw 0.5.0 CPU results](PERFORMANCE_SCENE_CPU.json) include median/p90, cold calls and pixel comparisons. Both current compiled backends stayed within one 8-bit RGB code in the sampled cases because of camera rounding. JPEG and sensor noise are disabled only for numerical validation outside timing. The warm benchmark includes the complete sampled capture.

## Uploaded Tesla T4 results, 0.4.0

The uploaded Colab run used Torch 2.11.0+cu130 on a Tesla T4. It ran 30 timed calls after eight warmup calls with cyclic backend order. These results precede the new paper model and cannot establish its GPU performance.

| Preset | B | Eager ms | Compiled ms | Compiled images/s |
| --- | --- | --- | --- | --- |
| fast | 1 | 44.6 | 48.3 | 20.96 |
| fast | 4 | 44.8 | 45.2 | 84.47 |
| balanced | 1 | 46.5 | 41.6 | 20.98 |
| balanced | 4 | 56.3 | 44.8 | 81.28 |
| full | 1 | 77.4 | 55.8 | 16.72 |
| full | 4 | 250.2 | 121.5 | 30.89 |

The separate full batch-4 profile measured 122.0 ms median and approximately 840 MiB of additional peak allocated memory for the compiled renderer. The higher-quality balanced supervision workload measured 68.4 ms median. The smaller-scene renderer plus tiny-CNN training workload measured 44.6 ms median. Their scene/antialias settings differ, so these two workloads are not a direct speed comparison.

The first full compiled call took about 14.7 seconds in that profile. [Historical T4 evidence](PERFORMANCE_T4_V040.json) preserves the runtime, original paired results, the one failed software test and corrected kernel-category trace counts.

## Measure the new version on your GPU

Use the [updated Colab notebook](https://colab.research.google.com/github/fmenegui/ecg-torch-renderer/blob/main/notebooks/colab_gpu_profile.ipynb). It compares current backends, runs the paper audit, profiles supervision and executes the U-Net example. The new paper model has not yet been timed on a GPU in this release.

```bash
ecg-renderer-compare --device cuda --presets fast balanced full \
  --batches 1 4 --steps 30 --warmup 8 --out comparison
```

## Compiled raster and scene

```python
render = ECGImageLayer(
    config=RenderConfig(height=384, width=512),
    compile_raster=True,
    compile_scene=True,
).to(device)
```

Raster compilation fuses deterministic coverage blocks. Scene compilation additionally fuses the material geometry, illumination and Newton updates. Both use `torch.compile(fullgraph=True, dynamic=True)`. Random sampling and scene orchestration remain eager, preserving the caller's generator sequence. Compilation may use additional memory and is opt-in. The eager path remains available.

Measure cold calls and steady-state throughput separately. Results depend on resolution, batch shape, Torch build, driver and hardware. A faster synthetic pipeline is useful only when its sampled distribution remains useful for the training task.

## Dense target memory

For a float32 mask tensor `[B,C,H,W]`, storage is `4 × B × C × H × W` bytes. This is retained tensor storage, not peak memory.

| Tensor at B=4, H=768, W=1024 | Approximate storage |
| --- | --- |
| RGB, 3 channels | 36 MiB |
| Validity, 1 channel | 12 MiB |
| Lead masks, 12 channels | 144 MiB |
| Instance masks, 15 channels | 180 MiB |
| One grayscale view, 1 channel | 12 MiB |

Camera warping, raster intermediates, the network and optimizer increase the peak. `ECGSupervisionLayer` omits baseline maps and does not return unrequested views. Boxes-only requests still need internal trace masks. Instance boxes need internal instance coverage, even when that coverage is not returned.

## Cost controls

| Control | Tradeoff |
| --- | --- |
| Resolution | Smaller batches of pixels reduce raster and network cost |
| `antialias_samples` | More subpixel samples improve raster coverage at added work |
| `preserve_extrema` | Retains narrow source extrema, adds envelope work |
| `lead_chunk_size` | Changes temporary memory and kernel workload |
| `surface_resolution` | Changes relief-field resolution |
| `inverse_iterations` | Coarse iterations, or output-resolution iterations in dense mode |
| `geometry_solver` | Coarse-to-fine uses two pixel refinements, dense provides a reference |
| `compile_scene` | Fuses deterministic geometry/lighting arithmetic, with initial compile cost |
| `shadow_samples` | Changes sampled shadow work |
| Requested supervision outputs | Changes dense-target storage and work |
| `surface_mode='off'` | Bypasses procedural surface deformation when no external maps are supplied |

Start with a modest resolution and the outputs your loss needs. Measure renderer-only latency and the full training step separately. More images per second only helps when the generated distribution and raster resolution remain useful for the task.
