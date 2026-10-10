# Performance

The hot path uses batched Torch operations with no image-file I/O, NumPy rasterization or CPU text drawing. GPU-friendly structure alone is not a measured throughput result. Benchmark your actual GPU and training objective.

## What is measured

The paired comparison below is **CPU only**, Torch 2.14.1+cpu, two threads, ten timed calls after three warmup calls. It compares the immutable 0.2.1 reference with the optimized image-only raster backend introduced in 0.3.0. It does not measure the new supervision outputs in 0.4.0.

{{performance_table}}

All six sampled CPU RGB comparisons had maximum absolute error zero. Compiled median speedup ranged from 1.07× to 2.72× in these cases. The eager backend regressed at balanced batch 4 and full batch 4. No CUDA speedup, clinical performance or real-photograph realism result is implied.

Download the [raw comparison JSON](PERFORMANCE_CPU.json), or inspect the [performance summary](https://github.com/fmenegui/ecg-torch-renderer/blob/main/PERFORMANCE.md).

## Compiled raster

```python
render = ECGImageLayer(
    config=RenderConfig(height=384, width=512),
    compile_raster=True,
).to(device)
```

This compiles deterministic coverage blocks with `torch.compile(fullgraph=True, dynamic=True)`. RNG, camera capture and scene assembly remain outside these blocks. The first call includes compilation or cache loading. Warm up separately and preserve the compiled cache between repeated local runs when appropriate.

Compiled throughput depends on image size, batch shape, driver, Torch build and hardware. Profile the steady-state workload. Compilation can use additional memory. The eager path remains available with `compile_raster=False`.

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
| `inverse_iterations` | Changes fixed inversion work |
| `shadow_samples` | Changes sampled shadow work |
| Requested supervision outputs | Changes dense-target storage and work |
| `surface_mode='off'` | Bypasses procedural surface deformation when no external maps are supplied |

Start with a modest resolution and the outputs your loss needs. Measure renderer-only latency and the full training step separately. More images per second only helps when the generated distribution and raster resolution remain useful for the task.
