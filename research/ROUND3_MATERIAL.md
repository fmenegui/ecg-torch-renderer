# Round 3: paper material, aging and liquid marks

Implemented `ecg_gpu/gpu_material.py` with nine independent contract and
spatial checks in `tests/test_material_r3.py`. This revision replaces the
appearance assumptions of a uniform age tint, Gaussian stain and perfectly
circular coffee ring with correlated material-coordinate fields.

## Physical interpretation

There are three distinct operations, all evaluated before camera warping.

| Output | Shape | Where it belongs | Interpretation |
| --- | --- | --- | --- |
| `substrate_factor` | B × 3 × H × W | Multiply linear paper reflectance before print | Oxidation, foxing, fiber density, thermal coating development |
| `deposit_transmission` | B × 3 × H × W | Multiply completed print in linear light | Colored liquid residue above paper and ink |
| `print_retention` | B × 1 × H × W | Multiply visible ink alpha before print compositing | Locally faded thermal image |

Neither absorption nor fading changes a trace mask, a physiological signal,
the page silhouette, or geometric visibility. A dark mark may make printed
ink harder to read while its geometric annotation remains unchanged.

Oxidation preferentially absorbs blue light, varies with broad moisture-like
history and reaches farther into different sheet edges by different amounts.
Foxing is a separately enabled, spatially clustered sparse process with
unequal cores and halos. Fiber texture is anisotropic and at most about one
percent in linear reflectance, rather than full-frame high-amplitude noise.

Coffee deposition has an anisotropic contact region, an irregular boundary,
nonuniform pinned sectors, weak interior deposition, and a secondary drying
front. Correlated fields span coarse, intermediate and fine material scales.
The absorbed stain has a broader asymmetric capillary footprint and a weak
tideline. Optional water marks use a much less chromatic attenuation.

Thermal fading is a smooth exposure footprint with local mottling and an
oblique drift. Optional `thermal_darkening_strength` models developed coating
in the background separately from fading of existing print. The model does
not infer pressure, temperature, exposure duration, paper chemistry or age
in calendar years from these dimensionless severity controls.

## Integration contract

Construct `PaperMaterial(height, width)` as a child module, then call
`factors = material(parameters, generator)` on the same device as rendering.
The existing vector controls are `paper_age`, `stain_x`, `stain_y`,
`stain_strength`, `coffee_strength`, `coffee_radius`, `thermal_fade`,
`fade_position`, and `fade_width`.

Additional vectors default to zero: `fiber_strength`, `foxing_strength`,
`water_strength`, and `thermal_darkening_strength`. Centers default to .5,
coffee radius to .15 and fading width to .22 when omitted. Radii use sheet
width units. Strengths use [0,1].

Recommended moderate inspection ranges are age .1–.35, stain .15–.65,
coffee .25–.8, fiber .3–1, foxing .15–.55 and water .2–.8. Extreme settings
are useful stress tests and should not occur in every ordinary acquisition.
Thermal development .1–.4 is already conspicuous. It defaults to zero.

Delete the old age, Gaussian stain, coffee annulus and thermal fade formulae
when enabling this material path. Do not apply both implementations.
Keep overall paper palette, grid ink, source paper textures and show-through
as separate acquisition controls. Apply the transmission before pen and
stamp compositing if those marks are intended to have been added after
the spill, or after them if their ink was already present during staining.

Zero strengths return exact multiplicative ones. To preserve historical
clean pixels, the renderer must select original sRGB values at neutral
locations, avoiding gratuitous sRGB → linear → sRGB round-trip drift.
The material generator consumes a fixed sequence of device RNG draws even
at zero strengths, maintaining stable within-version replay and avoiding
host synchronization in the hot path. Consequently activation changes the
rendering RNG stream relative to v0.7 and requires a new renderer fingerprint.
An inactive legacy path may skip this module entirely.

## Validation

Executed `python -m unittest tests.test_material_r3 -v` with CPU torch.
All nine tests passed. Tests check exact neutral factors, shape, positive
bounded attenuation, deterministic replay, batch diversity, blue-weighted
oxidation, stronger edge aging, irregular local coffee deposition, low
interior residue, stain centroid translation, separate thermal foreground
fading and background development, sparse foxing, subtle fibers, finite
strength gradients and absence of parameter mutation.

A six-panel inspection was rendered at `build/material-r3-review.jpg` with
schematic fixed gray print for new paper, aging, coffee, absorbed liquid,
foxing and faded print. This is a component inspection rather than evidence
of photographic indistinguishability or a measured real-data distribution.

## Primary references and limits

1. Deegan et al., *Nature* 389, 827–829 (1997),
   [Capillary flow as the cause of ring stains from dried liquid drops](https://www.nature.com/articles/39827).
   A pinned drying contact line and outward transport motivate stronger
   perimeter deposition. This is not a numerical solution of that flow.
2. Library of Congress,
   [The Deterioration and Preservation of Paper](https://www.loc.gov/preservation/care/deterioratebrochure.html).
   Paper composition and environment influence deterioration. This does not
   establish a universal age-to-color curve, which is deliberately absent.
3. Library of Congress,
   [Foxing and Reverse Foxing](https://loc.gov/preservation/outreach/tops/foxing/index.html).
   Foxing is treated as a distinct condition with complex modern-paper
   behavior. The implementation does not assert one microbial or chemical cause.
4. Koehler Paper,
   [Durability of Koehler Thermal Papers](https://www.koehlerpaper.com/media/docs/en/product-information/thermo/technische-infos/TH-Haltbarkeit-von-Koehler-Thermopapieren_GB.pdf),
   and Jujo Thermal,
   [Correct storing](https://www.jujothermal.com/technical-guide/correct-storing/).
   Manufacturer guidance motivates separating image stability from background
   development and treating light, heat, humidity and chemical contact as
   different histories. No storage-lifetime claims are encoded.

The fields are bounded appearance priors. They do not solve cellulose
oxidation, capillary transport, bidirectional scattering, coating reactions
or sheet mechanics. Real material scans and validation against photographed
clinical ECGs are still needed to calibrate a realistic sampling distribution.
