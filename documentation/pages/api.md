# API reference

This page derives signatures, dataclass defaults, layout names and parameter shapes from the installed package during the documentation build. Examples and semantic explanations are authored alongside that generated data.

## Public interfaces

{{api_signatures}}

## RenderConfig

{{config_table}}

`profile` chooses `clean`, `scan`, `photo` or `mixed`. `severity` controls the default augmentation strength. Individual parameter overrides still take precedence.

## SupervisionConfig

{{supervision_table}}

`outputs` selects advanced tensors. `leads` selects channel names/order. `binary` thresholds coverage. `visible_only` removes explicitly hidden pixels. See the [target contract](targets.html) for all shapes and box conventions.

## RenderBatch

| Field | Meaning |
| --- | --- |
| `image` | RGB `[B,3,H,W]` |
| `targets` | Dense and classification tensors selected by stage |
| `metadata` | Printed windows, lead IDs, units, sampled parameters and asset IDs |
| `reference` | Original signal tensor |
| `inverse_field` | Optional output-pixel-to-paper map |
| `canonical_image` | Optional pre-capture paper |
| `scene` | Optional normals, relief, illumination and visibility maps |

`stage='image'` omits task targets. `trace` requests per-lead traces and baselines by default. `geometry` requests grid/vertices, `metadata` requests labels/classes, and `all` combines them. `return_baselines=False` omits baseline maps. `return_instances=True` adds per-slot trace masks. `return_instance_boxes=False` skips legacy per-slot box extraction when another interface will calculate them.

## Rendering parameters

Shapes below use B as the batch axis and show the actual sample-parameter schema. Named category values are accepted for layouts, surfaces, table materials, shadow types and rhythm leads. Angles are in radians. Camera field controls are normalized; physical speed/gain use their indexed values.

{{parameters_table}}

## Console tools

| Command | Purpose |
| --- | --- |
| `ecg-render` | Render NPY/NPZ/WFDB signals and optionally export targets |
| `ecg-renderer-benchmark` | Renderer benchmark and examples |
| `ecg-renderer-gallery` | Procedural scene inspection gallery |
| `ecg-renderer-audit` | Public image-layer workload audit |
| `ecg-renderer-compare` | Current eager/raster/scene comparison, optional historical timing reference |
| `ecg-renderer-profile` | Detailed timing, operators, kernels and memory |

Run a command with `--help` for its current arguments. The documentation gallery is generated separately by `documentation/gallery.py` and `documentation/supervision_gallery.py`.
