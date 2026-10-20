# Template header collision, 0.9.1

An enlarged visual review of new ECG captures found overlapping speed/gain
text and the first template title in 0.9.0. The unit text used the legacy
height-relative position, while template titles used physical millimeters.
The existing tests checked visibility and calibration but not separation of
these two text blocks.

Template unit text now starts 1.5 mm from the material top. The first title
starts at least 5 mm from the top, with an additional gap when a template uses
larger typography. Both placements use the same canonical pixels/mm scale.
Default documents without a configured template retain their previous layout.
No waveform samples, trace geometry, labels or calibration values are changed.
Unit visibility still follows the actual glyph footprint after projection.

The regression test extracts the rendered unit ink and remaining title ink,
then checks disjoint vertical support. It covers two output widths, four
built-in/custom templates, three durations and all three available default
fonts. Before the correction, 54 of these 72 subcases failed. All 72 pass with
the correction. The complete suite and original binary fixtures also run in CI.

The generation review separately found limitations that this small fix does
not address: weak severe-crumple appearance, overly regular hole boundaries,
bitmap enlargement in macro views, dense 12x1 traces at 10 mm/mV and loss of
occluder appearance when reusing only geometry maps. These are model or recipe
limitations, not evidence of photographic equivalence.
