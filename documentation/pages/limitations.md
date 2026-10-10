# Scope & limitations

This package generates ECG images and paired targets. It does not contain a trained clinical digitizer, diagnose ECGs or guarantee recovery from arbitrary images.

## Geometry and realism

| Feature | Implemented scope |
| --- | --- |
| Paper deformation | Procedural 2.5D height fields with fixed-iteration inversion |
| Self-overlap | Can be supplied through externally rendered visibility/UV banks |
| Cloth dynamics | Not implemented |
| Phone/hand shadows | Analytic silhouettes |
| Lighting | Approximate local shading and sampled shadows |
| JPEG | Torch artifact simulation, not a complete codec |
| Handwriting | Procedural pen marks and supplied font styles |
| Table textures | Procedural defaults, optional authorized image banks |
| Visibility | Geometric and explicit occluder support, not readability confidence |

## Signal information

Cropping, occlusion and finite layout windows can remove information. Heavy blur and low contrast can make present information unrecoverable. The source reference still contains its original samples, so waveform losses must respect what the task actually observes.

The raster preserves narrow extrema through an envelope. At low resolution, that envelope can combine several samples into one pixel column. It does not make a many-to-one raster mapping invertible.

## Evidence boundaries

The included figures and integration runs use artificial ECG waveforms. The recorded performance comparison was on CPU. CUDA throughput, real-photograph realism and clinical waveform reconstruction have not been demonstrated in this release.

There is no validated claim that this generator is the most realistic in the world or that a model trained with it exceeds a commercial digitizer. Those claims require a reproducible comparative evaluation.

## Engineering boundaries

The renderer has no gradients to signals or rendering parameters. Python-level orchestration and randomization remain outside the optional compiled raster blocks. Cross-version/device bitwise reproducibility is not guaranteed. File I/O and initialization use CPU outside the hot path.

Use the [profiling tools](profiling.html) to identify your actual bottleneck, and the [validation guide](validation.html) to design evidence for your application.
