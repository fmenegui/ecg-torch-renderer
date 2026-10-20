# Audit corrections, 0.8.1

Audit base: `a4530aa0fa4b8dd6dbec47af2bbb84a3da893035`.
Date: 2026-10-20.

## Finding ledger

| Audit finding | Resolution in this change |
| --- | --- |
| 1. Orientation conflicts with absolute rotation | Recompose quarter overrides and derive the quarter from absolute angle. Reject explicit contradictions. |
| 2. Perspective mislabeled as missing units | Inverse-Jacobian pixel-area quadrature separates projected area from source coverage. |
| 3. Writable photo parameter has no effect | Per-record geometric gate implemented, plus static layer and CLI capture options. |
| 4. Retained damage in rectified outputs | Final documents apply substrate support. Explicit as_texture mode keeps albedo for 3D topology removal. |
| 5. Clipped fonts and annotation text | Actual-length atlas and static group bounds, configurable maximum font size and rejection of oversized overrides. |
| 6. Repeated RNG streams across replicas | Explicit checkpointed stream_id and stable offline source/view seed derivation. Independence is configured by the caller. |
| 7. Missing visible-fraction field | Instance outputs now include geometric ink fractions. Filtering is supported through training and grouped export. |
| 8. Disabled camera work still computed | Explicit static direct/sensor modes and JPEG opt-out. Recipes use specialized paths. Full mode intentionally retains heterogeneous controls. |
| 9. Inconsistent parameter domains | Integer construction controls validated. Lead/photo presence is binary. |
| 10. No external generalization evidence | Not fabricated. Controlled reference collection and paired ablation proposed in REALISM_ROADMAP.md. |
| 11. Limited paper mechanics | Still explicitly bounded. 3D/contact/bank extension proposed. |
| 12. Missing display simulation | Still absent. A dedicated emissive-display capture backend is proposed. |
| 13. Narrow artificial demo morphology | Different lead morphologies and exact limb voltage identities added. Still not a physiological or clinical validation dataset. |
| 14. Procedural visual cues | Template/material/scene improvements proposed with measured-reference gates. |
| 15. Outdated GPU evidence | Expose Mitsuba execution variant, remove global thread-count mutation, retain honest GPU evidence boundary. Fresh target-GPU timing still requires hardware. |
| 16. Geometric validity versus readability | Contract clarified. Ink coverage fractions are geometric and do not promise readable/recoverable signal. |

## Public contracts

- `angle` is absolute radians. Its nearest quarter-turn determines the coarse
  orientation class. At exact half-quarter ties, Torch round-to-even applies.
- `quarter` alone keeps sampled angular jitter and recomposes total rotation.
- `parameters['photo']=0` disables geometry for that record. Sensor behavior is
  controlled independently. Formal `photo=False` selects a rectified document.
- `as_texture=True` requires `photo=False`. It prepares intact albedo and
  separate material support for 3D clipping. A damaged final document cannot
  silently be passed to the path tracer as an intact texture.
- `instance_visible_fraction` integrates surviving canonical-page waveform ink,
  not cell area, time samples or optical confidence. Pixel quadrature is
  resolution-dependent, particularly for tiny structures and discontinuous
  externally supplied UV maps.
- `camera_mode='direct'` requires sensor_active zero. Sensor mode requires one.
  Full mode retains arbitrary per-record blends and computes both paths.
  `jpeg_enabled=False` requires jpeg_active zero. Defaults follow static settings.
- Large custom fonts must fit max_font_size, including the header scale of 1.15.
  This bound sizes glyph work at construction and avoids device scalar reads.
- Reproducible independent workers/ranks need distinct stream_id values. Stable
  per-source/view capture seeds can instead ignore worker assignment.
- Rectified missing paper is displayed on a white background and excluded from
  visible training targets. Projected/amodal masks remain available separately.

## Compatibility

The waveform raster kernels, lead identity and time/voltage scale are unchanged.
Historical binary raster fixtures are preserved. New source/configuration fields
change renderer signatures. Recreate the same version and configuration to load
checkpoints, rather than assuming cross-version pixel replay. The new demo signal
also intentionally changes galleries generated from the artificial example.

Examples preparing offline 3D input now explicitly pass as_texture=True.
Existing callers that supplied inconsistent quarter/angle pairs must correct
them. Fractional lead/photo presence is rejected rather than silently coerced.

## Verification

The audit regressions exercise actual rotations, analytic projective containment
and area, missing substrate, large-font support, fractional visibility filtering,
checkpointed streams, static camera pixel/RNG parity, CLI target export and limb
voltage identities. The final suite and smoke results are recorded in
AUDIT_FIXES_VERIFICATION.json after execution.

The original binary fixture exists upstream but could not be materialized through
the current connector. Local runs disclose the two unavailable fixture tests.
The branch CI uses a full checkout and must run those tests without replacing
or regenerating their expected data. CUDA and external real-image experiments
are not claimed from CPU tests or from successful integration alone.
