# Architecture and engineering contracts

The physiological input is immutable reference data. The renderer converts it
into a document, projects the document and observes the scene. Geometric targets
follow the same projection and physical visibility. Optical blur, noise and
compression affect RGB without rewriting geometric ownership.

## Component responsibilities

| Component | Owns | Does not own |
| --- | --- | --- |
| `DocumentTemplate` | Physical margins, typography, fictional headers, colors | Signal morphology or diagnoses |
| `PaperMaterial`, `ThermalPrint`, `MaterialHistory` | Substrate, coating, deposition and a declared history | Camera noise or independent label warps |
| `MeasuredPaperBank`, `SceneBank` | Construction-time decoding, scale and provenance | Runtime file access |
| `PaperSurface`, `DocumentScene` | Shared mesh, normals, inverse UV, shading and visibility | Source signal edits |
| `CameraAppearance`, `PhysicalCamera` | Linear optics, sensor observation, codec | Geometric trace masks |
| `EmissiveDisplay` | Pixel apertures, emission, PWM, glass | Flexible paper mechanics |
| `paper_mechanics`, `pathtrace` | Offline geometry, contact, explicit edges and primary rays | Training-loop allocations or CUDA claims |
| `ECGRenderer` | Ordering and composition of those stages | General fluid or cloth physics |
| `ECGImageLayer`, supervision and training adapters | RNG replay, target selection and network integration | Implicit patient splits |

## Design decisions

The 0.9 additions use composition and small validated configuration dataclasses.
The renderer remains an orchestrator with legacy branches, rather than a newly
rewritten framework. This limits regression scope. It is not a claim that every
legacy method is already minimal or that all possible SOLID interpretations are
satisfied. Further extraction should be driven by real extension needs.

| Principle | Concrete application and verification |
| --- | --- |
| Single responsibility | Optics, display emission, thermal print, import and offline mechanics live in separate modules. Physics tests exercise them independently. |
| Open/closed | Camera subclasses implement `scene_depth`, `forward` and `specification`. Renderer dispatch uses the contract without concrete camera-type branches. |
| Liskov substitution | Camera extensions retain sRGB input/output, shape, device and optional depth semantics. A substitute camera is tested end to end. Incompatible modes are rejected during construction. |
| Interface segregation | Runtime model calls consume tensors. File/manifest loaders and NPZ export are separate construction/offline operations. Optional maps remain optional. |
| Dependency inversion | A camera model is injected into `ECGRenderer`. Its configuration enters the renderer specification and checkpoint identity. The orchestrator does not construct external calibration from hidden global state. |

Models are `nn.Module`s with registered buffers and no trainable parameters.
`to(device)` moves their state once. Validated immutable configs select optional
work at construction. The training forward path uses torch on the input device,
without NumPy, file I/O, CPU copies or explicit tensor scalar extraction.
Offline simulation and provenance hashing intentionally use CPU and files.

The supplied generator remains caller-owned. Thermal print and calibrated paper
crops reuse existing per-record phase values so enabling them does not consume
extra geometry random draws. Checkpoints hash component specifications, physical
profiles, asset content and provenance. A changed extension is rejected on load.
Version 0.9 changes that specification schema content. Resume old 0.8 checkpoints
with the old package, or construct and validate a new run explicitly.

## Numerical and performance contracts

Coordinates use pixel centers with `grid_sample(align_corners=False)`. A single
canonical pixels/mm scales both voltage and time. Banks preserve material UVs.
Visibility is applied once, with the inverse Jacobian where material-area
fractions are required. Frame and bezel coverage do not own lead identity.

Defocus accumulates one blurred image at a time, without a five-level full-image
stack. Display quadrature has a fixed construction-time bound of 1..4 per axis.
JPEG can be omitted statically. Expensive sheet geometry is prepared offline and
cached as maps. Material scans reject missing physical coverage. These are
bounded-cost mechanisms, not universal throughput guarantees.

The new benchmark script records warmup, repeat count, median/min/max latency,
image size, batch size, stage, threads, software and device. Compare costs on the
same machine and output configuration. No CUDA performance evidence was collected
in this CPU workspace. CUDA device tests remain conditional.

## Quality gates

`python -m unittest discover -s tests -v` is the complete contract suite. It
includes immutable raster fixtures, independent physical calculations, projected
supervision, checkpoint replay, CLI, adversarial geometry and training adapters.
The optional job adds actual Mitsuba primary-ray tests and one YOLO training
epoch without pretrained weights. The main training example performs optimizer
steps through a downstream image model.

CI tests Python 3.10/Torch 2.6 and Python 3.12/3.13 with current Torch. Quality CI
adds fatal-error linting across the codebase, stricter linting for new modules,
branch-aware coverage with an 85 percent aggregate floor for the seven new model
modules and an uploaded coverage report. Coverage measures exercised paths, not
physical accuracy. It is not a claim of 100 percent project coverage.

The docs build derives signatures and config tables from the installed package.
Validation checks code syntax, links, figure hashes and API source hashes. Tests
use small analytical references such as image-plane thin-lens geometry, interval
intersection for PWM exposure, discrete free fall and closed-slab volume.

Visual review exports actual captures and inspects them separately from unit
results. It exposed a display-aliasing defect during 0.9 development, leading to
analytical aperture integration and a distant-panel neutrality regression test.
Real-image generalization requires a different dataset and remains unverified.

## Primary references

- Macklin, Müller and Chentanez, [XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf), 2016. The implementation uses compliant distance constraints with rest-shape bending proxies.
- Pharr, Jakob and Humphreys, [Cameras and Film](https://pbr-book.org/4ed/Cameras_and_Film), PBRT fourth edition. The fast model is a restricted thin-lens/RGB approximation.
- [Mitsuba emitter reference](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_emitters.html). Offline light transport remains optional.
