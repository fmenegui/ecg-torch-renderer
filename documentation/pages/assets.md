# Custom assets & layouts

Extend the generator at construction time. Banks are registered as module buffers, hashed for provenance, then used on-device during rendering.

## Custom layout

```python
from ecg_gpu import ECGRenderer, LayoutSpec, RenderConfig

layout = LayoutSpec(
    name="focused",
    rows=(("I", "II", "III"), ("V1", "V2"), ("V3", "V4", "V5", "V6")),
    rhythm=("II", "V1"),
)
renderer = ECGRenderer(RenderConfig(), layouts=[layout])
```

Rows can have different column counts. Short rows require unique standard leads. Rhythm strips can repeat leads, up to three strips. `-aVR` prints inverted aVR with canonical identity metadata.

## Photo and texture banks

```python
# Supply authorized RGB textures with values in [0,1].
paper_bank = torch.rand(4, 3, 128, 128)
table_bank = torch.rand(4, 3, 128, 128)
renderer = ECGRenderer(
    paper_textures=paper_bank,
    backgrounds=table_bank,
)
```

These random tensors only illustrate the bank format. In an experiment, use authorized texture images without patient information. Load and resize the bank before training. Large banks consume device memory after `.to(device)`.

## Fonts and annotations

```python
renderer = ECGRenderer(
    fonts=["/path/to/authorized-font.ttf"],
    artifact_texts=("COPY", "ARCHIVE", "TEST ECG"),
)
```

Glyph rasterization happens once during construction with Pillow. The generated atlas becomes a device buffer. Default annotations are printed text and procedural scribbles. Handwriting styles require a suitable supplied font or a richer custom augmentation.

## Externally rendered document geometry

For fully 3D paper, self-overlap or offline path-traced shading, provide a bank of camera-consistent inverse maps. Every map must describe the same camera and paper coordinates.

| Asset | Shape | Meaning |
| --- | --- | --- |
| `inverse_field` | `[K,H,W,2]` | Output pixel → paper UV, normalized for `align_corners=False` |
| `mask` | `[K,1,H,W]` | Visible paper support in `[0,1]` |
| `normals` | `[K,3,H,W]` | Camera-frame normal vectors |
| `height` | `[K,1,H,W]` | Nonnegative relief in half-page-width units |
| `shading` | `[K,1,H,W]` | Optional nonnegative illumination multiplier |

```python
from ecg_gpu import DocumentAssets

assets = DocumentAssets.from_npz("authorized-paper-maps.npz")
renderer = ECGRenderer(document_assets=assets)
```

The NPZ requires `schema="ecg-document-assets-v1"`. Generate UV and visibility offline with your own geometry tool, then package them together. The library resamples their maps on-device. It does not run Blender, simulate cloth or generate a 3D mesh bank for you.

Normals must be nonzero on visible paper and are normalized at construction. Shapes, finite values, visibility range and nonnegative height/shading are validated. Do not include document text or patient information in these assets.
