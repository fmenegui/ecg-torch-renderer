# Renderer research and engineering workflow

This project generates ECG documents and capture artifacts from supplied signals.
The physiological signal is the reference. Do not synthesize diagnoses or alter
the reference waveform as a shortcut to photorealism.

## Parallel roles

When the user requests a swarm or parallel agent work, assign disjoint files and
use a coordinating integrator. The current role definitions are:

| Role | Responsibility | Required evidence |
| --- | --- | --- |
| Research | Primary literature, device conventions and realism gaps | Sources and falsifiable acceptance criteria |
| Signal/document fidelity | Lead identity, polarity, timing and physical calibration | Independent analytical checks |
| Capture realism | Paper, optics, lighting and sensor observations | Physical reference checks and rendered examples |
| Performance | Allocations, profiling and exact output parity | Reproducible timings on named hardware |
| Independent QA | Adversarial cases, regressions, package and CLI | Test log, explicit failures and limitations |
| Integrator | Reconcile changes, inspect images and publish | Final reviewed commit and validation report |

Review agents may request fixes from implementers. Never infer that a passing
unit suite proves photographic realism or downstream clinical generalization.

## Acceptance gates

- Preserve original mV waveform, lead ownership, sequential time windows and
  the shared pixels/mm calibration in both axes.
- Keep rendering and supervision in torch on the input device. No CPU/file
  work, tensor `.item()` or NumPy in the training hot path.
- Geometry, occlusions and validity must remain aligned across image and targets.
  Optical blur changes the observation, not the geometric trace label.
- Test a change against an independent reference when claiming a physical law.
  Run `python -m unittest discover -s tests -v` for integration.
- Export and inspect a multi-layout adverse-capture gallery using
  `python examples/capture_scenarios.py --out build/scenarios`.
- Report CUDA checks as unavailable when no GPU exists. CPU speed measurements
  must not be advertised as GPU measurements.
- Separate algorithmic assertions, human visual review and external real-image
  benchmark results. Record unavailable evidence explicitly.
- Preserve upstream binary regression fixtures. Do not regenerate expected
  fixtures from modified code to make a failure disappear.

Current review evidence and unfinished external validation are in `research/`.
