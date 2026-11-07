# Capture & print artifacts

## Static camera paths in 0.8.1

`RenderConfig(camera_mode='direct', jpeg_enabled=False)` omits optical sensor
and JPEG work. The `sensor` path omits the direct alternative. These settings
are structural, so no device-to-host scalar inspection is needed per batch.
`full` retains mixed per-record sensor controls and computes both alternatives.
Scenario recipes choose compatible settings automatically. Pixel and RNG
parity are checked against the full path for each specialized domain.

Direct mode requires `sensor_active=0`, sensor mode requires one, and a disabled
codec requires `jpeg_active=0`. Conflicting overrides raise rather than being
silently ignored. Optics do not change geometric waveform targets.

## Material damage in 0.8.0

Paper appearance is constructed in material coordinates before perspective or
folding. Oxidation and fiber density affect unprinted reflectance. Deposited
liquid absorbs linear light after printing and annotations. Thermal fading
reduces observed print contrast, while thermal darkening develops the paper
coating. These processes preserve geometric trace targets.

| Controls | Meaning |
| --- | --- |
| `paper_age`, `foxing_strength` | Irregular oxidation and clustered pigment spots |
| `stain_strength`, `water_strength` | Absorbed stain and drying front |
| `coffee_strength`, `coffee_radius`, `stain_x`, `stain_y` | Irregular contact line and uneven dried deposit |
| `fiber_strength` | Subtle anisotropic substrate detail |
| `thermal_fade`, `fade_position`, `fade_width` | Mottled loss of observed thermal-print contrast |
| `thermal_darkening_strength` | Local developed coating independent of print fading |
| `tear_active`, `tear_count`, `tear_raggedness` | Up to four irregular open edge tears |
| `edge_wear_strength`, `edge_wear_depth` | Shallow abrasion of outer boundaries |
| `hole_active`, `hole_center`, `hole_radius`, `hole_angle` | Three independent irregular punctures |
| `cut_active`, `cut_center`, `cut_length`, `cut_width`, `cut_angle` | Three finite open slits |
| `missing_corner_active`, `missing_corner_index`, `missing_corner_size` | One removed corner |

Strengths are dimensionless augmentation controls, not elapsed years or fluid
concentration. Damage centers use sheet coordinates in `[-1,1]`, while stain
centers use `[0,1]`. Radii, widths and
lengths use half-page-width units in both axes, so `.01` means 1.5 mm on a
300 mm sheet. Right and bottom box edges remain exclusive.

Missing material removes the printed image, trace targets and shadow caster
together. A stain does not become a hole. `damage_detail="edge"` evaluates only
the first edge tear and rejects enabled advanced damage. The default
`"rich"` supports every damage type at higher compute cost.

`capture_scenario("missing_fragments")`, `"cut_sheet"`, `"water_damage"`,
`"coffee_spill"`, `"thermal_damage"` and `"oblique_phone"` provide complete
recipes. The perspective transform rejects a horizon crossing instead of
silently dividing by a near-zero projective denominator.

![Irregular material damage, liquid history and thermal development](assets/round3-gallery.jpg)

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

As of 0.6.0, motion blur integrates a finite, uniform, constant-velocity shutter
path convolved with Gaussian defocus. The normalized kernel has 9×9 pixel
support. Very large user overrides are truncated by that support. Read and
shot noise combine as variance `read_noise**2 + shot_noise**2 * intensity`.
This is a Gaussian sensor approximation, not an electron-counting model.

In 0.7.0 exposure, optical filtering, shot/read variance, geometric resampling,
surface lighting and compositing operate in linear light. Highlights are
retained until sensor saturation. Digital white balance follows sensor noise.
The bounded 9×9 kernel accepts `blur` in [0,1.1] and motion in [0,3.2] output
pixels. Larger kernels require an explicit alternative optical model.
Paper-attached pen and stamp coverage now follow the sheet's material warp.
The separate opaque rectangle remains a camera/scene occluder.

Optional local measured backgrounds can be loaded once with `load_rgb_bank`.
For actual 3D area-light transport and ray-visible supervision, install the
optional Mitsuba backend and use `examples/pathtraced_capture.py`. Its CPU
offline work is separate from rendering inside a torch training loop.

Seven coherent recipes are available through `capture_scenario(name)`, returning
a configuration and parameter dictionary for the usual image or supervision
layer. Names are `clean_print`, `flatbed_scan`, `aged_scan`, `phone_photo`,
`low_light_phone`, `folded_photo` and `damaged_archive`. Scanner recipes disable
camera perspective, motion and directional scene lighting. Severity is a stress
control, not a calibrated probability of unreadability. Geometric validity
does not imply optical readability.

The existing 125-figure gallery is retained from 0.5.0. Current 0.6.0 scenario
outputs and their reproducible recipes are generated by
`python examples/capture_scenarios.py --out build/scenarios` and summarized in
the repository's `research/SWARM_REPORT.md`.

```python
from ecg_gpu import capture_scenario, ECGSupervisionLayer
config, parameters = capture_scenario('folded_photo', severity=.8)
layer = ECGSupervisionLayer(config=config, parameters=parameters)
```

## Paper and print

Control ink width, color, density, thermal fading, scan bands, grid styles, grid phase, missing lead slots, label placement, fonts, unit headers, paper tint/age/noise, stains, coffee rings and reverse-side bleed.

![Local thermal fading of printed traces](assets/gallery/thermal-fade.jpg)

## Explicit occlusion

Tears remove part of the paper. Opaque rectangles, procedural pen strokes and rotated artifact text cover the printed page. The `valid` mask accounts for these explicit visibility changes.

![A torn paper edge](assets/gallery/tear.jpg)

Shadowed or blurred pixels remain geometrically valid. Visibility does not mean the trace is readable. Keep a separate quality criterion if your application needs it.

`photo=False` produces a canonical page and disables the camera geometry, surface deformation and external document warps. It is useful for rectified training targets. It does not automatically remove every configured print artifact.



## Scanner modes

`RenderConfig(scan_modes=True)` adds flatbed scanner observation to records captured without photographic geometry. `scan_mode` selects color (0), grayscale (1) or bilevel black-and-white (2), sampled at 50/30/20 %. `scan_dpi` (log-uniform 72–600) sets the scanner resolution: when it is below the page's pixel density, each scanner pixel is integrated and held, as in an upscaled low-resolution scan. `scan_threshold` (0.45–0.70) is the bilevel luminance cut. Targets are unchanged.

```python
from ecg_gpu import ECGImageLayer, RenderConfig

layer = ECGImageLayer(fs=500, config=RenderConfig(profile="scan", scan_modes=True),
                      parameters={"scan_mode": 2, "scan_dpi": 150.}, photo=False)
```

## Phone processing

`RenderConfig(phone_isp=True)` adds effects that phone photographs of printouts show and the base camera lacks, only on photographed records: lateral chromatic aberration (`ca_px`), sharpening halos (`sharpen_amount`, `sharpen_sigma`) and specular glare from glossy thermal paper or a flash (`glare_strength`, `glare_x`, `glare_y`, `glare_radius`). The stage runs after the base camera, so it approximates rather than reproduces a phone's processing order. Targets are unchanged.

## Reprints, photocopies and faxes

`RenderConfig(print_technology=True)` reproduces the page before it is photographed or scanned. `print_tech` selects an original print (0), a laser reprint whose colored grid is halftoned with an ordered-dither screen (1, `halftone_cell_px`), a grayscale photocopy with `copy_generations` passes of blur, contrast, toner speckle and background lift (2), or a fax printout, bilevel at 204 x 98 dpi (3). Targets are unchanged.
