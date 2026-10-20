# Physical realism extensions

Version 0.9 adds optional models. The default pipeline retains the previous
rendering behavior. None of the shipped parameter presets is a measured camera
or paper calibration. External real-image validation remains pending.

![Actual synthetic ECG outputs from the optional models](assets/realism-v090-gallery.jpg)

Artificial ECG inputs. Renderer outputs, not AI-redrawn waveforms.

## Compose the training renderer

```python
from ecg_gpu import (
    ECGImageLayer, RenderConfig, ThermalPrintConfig,
    PhysicalCamera, LensConfig, SensorConfig,
)

layer = ECGImageLayer(
    fs=500,
    config=RenderConfig(
        width=640, height=480, profile='photo', camera_mode='sensor',
        jpeg_enabled=False, jpeg_probability=0,
    ),
    document_template='clinical',
    thermal_print=ThermalPrintConfig(),
    camera_model=PhysicalCamera(
        LensConfig(focus_distance_m=.38, camera_distance_m=.38),
        sensor=SensorConfig(full_well_electrons=12000, read_noise_electrons=2),
        jpeg_enabled=False,
    ),
)
# Move once with layer.to(device), then pass supplied [B,12,T] mV signals.
```

These components add torch operations on the input device. They do not change
reference samples. Noise and blur change observed RGB, while masks retain
geometric coverage. A very blurred trace can have a valid geometric label and
still be unreadable. Use a separate readability criterion for such training data.

## Document and print

`DocumentTemplate` defines margins and font sizes in millimeters, fictional
header lines and RGB colors. Built-in names are `clinical`, `archive` and
`compact`. They are generic conventions, not replicas of commercial equipment.
Speed, gain and time windows still come from the actual signal and parameters.
Templates raise when the sheet has no usable plot height or typography exceeds
`max_font_size`. Increase that construction-time bound for unusually short
recordings or large physical fonts. A configured template controls font size
and document colors. Legacy random controls for those properties no longer
select the template's design.

`ThermalPrint` adds column-correlated head defects, feed bands with a period in
millimeters, local retention and directional substrate reflectance. It reuses
an existing per-record phase without consuming the geometry RNG stream.
A head bank covers at most 8191 elements at `head_pitch_mm`. Larger physical
sheets are rejected. This is an illustrative print prior, not measured thermal
printer hardware or raised cellulose microgeometry.

## Paper scans at physical scale

`MeasuredPaperBank.from_json(path)` reads this manifest at construction time:

```json
{
  "schema": "ecg-paper-scans-v1",
  "scans": [{
    "path": "paper-albedo.png",
    "pixels_per_mm": 8,
    "source": "authorized flat-field scan, collection A",
    "license": "collection owner's permission",
    "kind": "measured"
  }]
}
```

Scans must have the same dimensions and scan resolution. Supply flat-fielded
sRGB albedo without directional illumination, ECG content or patient data.
`kind` is the caller's declaration, not proof of measurement. Source, rights,
scale and SHA-256 enter the renderer specification. Canonical pixels/mm sets
the crop scale. Requests larger than the physical scan raise. Images are never
silently repeated or stretched. Existing `paper_textures` remain available for
uncalibrated decorative texture banks. Choose one source.

## Lens and sensor

`PhysicalCamera` requires `camera_mode='sensor'` and a matching JPEG setting.
For focal length f, focus distance s, object distance z and f-number N, the
circle-of-confusion diameter on the sensor is:

```text
c = f² |s - z| / [N (s - f) z]
```

Lengths are converted to meters and sensor pixels using the declared effective
pixel pitch. Output resolution/downsampling must agree with that effective
pitch. The Gaussian PSF sigma is c/4, matching the second moment of a uniform
blur disk, up to finite-kernel discretization error. Interpolation in variance uses a finite bank at 0, .5, 1, 2 and 4 pixels. Requested
sigma beyond `max_sigma_pixels`, at most 4, raises instead of clipping the blur.
The existing residual PSF and motion exposure remain additional observation
controls. Their convolution adds blur, so set them deliberately when fitting.

Scene relief converts from half-page-width units using canonical pixels/mm.
Distances refer to a fronto-parallel plane. Background depth uses the same
reference plane. This is a bounded RGB approximation, without lens visibility
integration at depth discontinuities, a CFA, diffraction or a fitted ISP.

For full well F electrons and read noise r electrons, normalized noise variance
is x/F + (r/F)^2 before white balance. Shot noise uses the Gaussian approximation
already present in the sensor pipeline. Low-count exact Poisson behavior is not
modeled. With no `SensorConfig`, the original batch noise controls remain active.

`PhysicalCamera.from_json` accepts `ecg-camera-calibration-v1` with `lens`,
optional `sensor` and `provenance`. The provenance `kind` must be `illustrative`
or `measured`. A measured declaration also requires `source`, `method` and the
measurement dataset's lowercase `data_sha256`. The file itself is hashed. These
fields establish traceability, not independent calibration certification.

## Emissive monitor captures

```python
from ecg_gpu import ECGImageLayer, RenderConfig, DisplayConfig

screen = ECGImageLayer(
    config=RenderConfig(
        width=640, height=480, profile='photo', surface_mode='off',
        camera_mode='sensor',
    ),
    document_template='compact',
    display=DisplayConfig(
        native_width=1280, native_height=800, subpixel_order='RGB',
        duty_cycle=.55, refresh_hz=120, exposure_s=.0012,
        readout_s=.018, footprint_samples=2,
    ),
)
```

A display is a rigid emissive panel. Paper assets, thermal print and flexible
sheet geometry are rejected for this medium. Perspective and radial distortion
remain available. Panel pixels hold sampled document content. RGB/BGR apertures
are integrated analytically within stratified local box footprints, reducing
false rainbow aliasing when the panel is unresolved. Rotated footprints use
axis-aligned local bounds, which can smooth more than an exact polygon integral.

Temporal bands use the exact exposure integral of a rectangular full-panel PWM
backlight with row-dependent shutter start times. A continuously lit display,
`duty_cycle=1`, has no refresh bands. This is not frame scanout, LCD response,
OLED geometry or rolling shutter distortion of moving content. The report is
static. Glass uses Schlick reflectance with explicit view cosine and a softbox
prior. That view cosine is not recovered from the projective warp. Reflections
can also be supplied as linear RGB to `EmissiveDisplay.forward`.

The bezel is outside the printed panel UV and receives no waveform labels.
Paper illumination and object shadows do not darken the emitted panel.
Annotations supplied through pen/stamp controls are treated as digital content.
The dedicated opaque camera occluder still blocks both RGB and targets.

## Liquid and storage chronology

`MaterialHistory` couples spill volume, elapsed minutes, humidity, solute,
storage age, UV dose and heat exposure. Wetness decays, deposits accumulate and
residual local cockling remains. Deposition and geometric displacement share
the same material center and radius. Normals, inverse projection, shadows and
labels are recomputed from the changed surface. This requires enabled paper
geometry. Rectified texture export omits the geometric warp.

Coefficients are illustrative. This is not a fluid solver, capillary transport
simulation, measured thermal chemistry or inference of real storage conditions.
Explicit parameter overrides can replace history-derived values for experiments.

## Offline sheets, edges and geometry banks

```python
from ecg_gpu.pathtrace import PathTraceConfig, pathtrace_capture, capture_geometry_assets
from ecg_gpu.paper_mechanics import SheetConfig

capture_config = PathTraceConfig(
    mesh_columns=33, mesh_rows=25,
    sheet=SheetConfig(iterations=72),
    surface='multifold', edge_tear=.1,
    paper_thickness_m=.00006, edge_fibers=24, cut_lip_lift_m=.00015,
)
# canonical = a single rectified stage='all' RenderBatch with instances,
# prepared with photo=False, as_texture=True.
# capture = pathtrace_capture(canonical, config=capture_config, return_geometry=True)
# bank = capture_geometry_assets(capture)
# bank.to_npz('geometry-bank.npz')
# Pass DocumentAssets.from_npz(...) as document_assets in the training renderer.
```

`SheetConfig` enables offline XPBD distance constraints, gravity, damping and
unilateral table contact. Two-hop rest-shape constraints retain prescribed
crease memory. They are a bending proxy. Solver inverse masses are one in
arbitrary solver units. Compliance is not a measured paper modulus. This is not
a full thin-shell constitutive law, plastic crease evolution or tear propagation.
The solver bounds vertex count at 2048 and rejects unresolved material strain.

Nonadjacent triangle intersections, including coplanar overlap, are checked and
rejected. There is no general self-contact response or continuous collision
solver. Shared-vertex neighbors are excluded from that test. Offset shells can
still require more detailed validation for extreme curvature. Table penetration
from finite thickness is rejected. Increase clearance or reduce thickness.

The printed front remains unchanged when adding thickness. Separate unprinted
back and sidewall shapes close all damage boundaries. Fibers are small 3D
cylinders attached to the same boundary. Cut lips are a prescribed local lift,
with a final 5 percent edge-strain gate. Labels come from actual camera rays and
stop on missing paper, edge shapes, fibers and physical occluders.

Exported geometry banks contain center-ray UV, visibility, camera-frame normals
and nonnegative world relief. They contain no document bitmap or patient data.
Maps retain camera/configuration and mesh hashes. Baked perspective is part of
the bank. Additional runtime transforms compose with it. Center-ray binary
visibility differs from subpixel-averaged labels at silhouettes. Do not average
UV across foreground/background boundaries. Shading is recomputed at runtime,
so this bank does not preserve all baked light transport or occluder appearance.
NPZ writes are validated and atomic.

## Scene assets and honest evaluation

`SceneBank.from_json` reads `ecg-scene-bank-v1`, with `assets` containing `path`,
`source`, `license`, `family` and `split`. Split values are `train`, `validation`
and `test`. The loader rejects identical image bytes and asset families shared
across splits. Use the selected bank as `backgrounds=bank`. Metadata declarations
do not establish legal rights automatically. Near-duplicate detection and
patient/device grouping still require dataset curation.

Run `python examples/physical_realism.py --out build/realism --pathtrace --benchmark`
for actual outputs, a geometry bank and named-hardware timings. Run
`python examples/capture_scenarios.py --out build/scenarios` for adverse layouts.
These demonstrate algorithm behavior and visual plausibility. A held-out,
authorized real-print, camera and display collection is still needed to measure
photographic similarity and downstream digitization accuracy. See the research
roadmap for the acquisition protocol.
