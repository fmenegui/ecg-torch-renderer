# Capture & print artifacts

Printing, geometry and capture are independently configurable. Their default distributions are a starting point, then should be adapted to the images you expect to digitize.

## Geometry

Perspective, small rotations, quarter turns, zoom, translation, crops, radial lens distortion, curl and creases alter the sampling field. Targets use the same field.

![A partially cropped ECG page](assets/gallery/crop.jpg)

```python
render = ECGImageLayer(parameters={
    "angle": Uniform(-.2, .2),
    "quarter": Choice([0, 1, 2, 3]),
    "perspective_x": Uniform(-.2, .2),
    "perspective_y": Uniform(-.12, .12),
    "radial": Uniform(-.03, .03),
})
```

## Camera

| Control | Effect |
| --- | --- |
| `blur` | Defocus component of the point-spread function |
| `motion`, `motion_angle` | Directional motion blur |
| `read_noise`, `shot_noise` | Additive and intensity-dependent sensor noise |
| `exposure`, `white_balance`, `gamma` | Tone and channel response |
| `vignette` | Spatial attenuation near the frame edge |
| `jpeg_quality`, `jpeg_active` | 4:2:0 chroma and 8×8 DCT artifact simulation |

![Directional motion blur](assets/gallery/motion.jpg)

![Low-quality JPEG artifact simulation](assets/gallery/jpeg.jpg)

JPEG here is a decoded-artifact approximation implemented in Torch. It does not write an entropy-coded JPEG file. The CLI separately encodes the final image if you choose `.jpg`.

## Paper and print

Control ink width, color, density, thermal fading, scan bands, grid styles, grid phase, missing lead slots, label placement, fonts, unit headers, paper tint/age/noise, stains, coffee rings and reverse-side bleed.

![Local thermal fading of printed traces](assets/gallery/thermal-fade.jpg)

## Explicit occlusion

Tears remove part of the paper. Opaque rectangles, procedural pen strokes and rotated artifact text cover the printed page. The `valid` mask accounts for these explicit visibility changes.

![A torn paper edge](assets/gallery/tear.jpg)

Shadowed or blurred pixels remain geometrically valid. Visibility does not mean the trace is readable. Keep a separate quality criterion if your application needs it.

`photo=False` produces a canonical page and disables the camera geometry, surface deformation and external document warps. It is useful for rectified training targets. It does not automatically remove every configured print artifact.
