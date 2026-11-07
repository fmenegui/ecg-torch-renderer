# Visual realism assessment, 0.12.0

Scope: a human-style visual review of committed renderer outputs plus a code
review of the models behind them. No new images were rendered (the review
environment had no PyTorch), and no quantitative comparison against real
images was run. Several committed galleries predate 0.12.0, so some findings
may describe older defaults. Each item names its evidence so it can be
rechecked on a fresh gallery.

Reviewed images: `research/assets/realism-v090-gallery.jpg`,
`round3-gallery.jpg`, `round2-scenarios.jpg`, `round3-pathtraced-gallery.jpg`,
`round3-supervision-rgb.png`, `supervision-rgb.png` and
`docs/assets/gallery/speed-{0,1,2}.jpg`. Real-world reference: the artifact
taxonomy of ECG-Image-Database (Shivashankara et al., arXiv 2409.16612) and
public photographs of printed 12-lead ECGs.

## Overall

The offline path-traced backend is the most convincing output: folds bend the
grid and traces, shading follows geometry and paper has visible thickness.
The torch training path reads as a clean vector rendering with overlays.
Grids and traces stay straight under "crumpled" and "folded" recipes, damage
looks pasted on, and the page layout and text vary far less than real
printouts. This matches the measured grid-point gap (online-trained F1 0.916
on torch images, 0.025 on ray-traced images).

## Findings

Severity reflects the expected effect on a model trained on these images and
applied to real captures.

### High

1. **Paper speed changes page size instead of time window.** `_instances`
   sets `ppm = (W-1) / (duration * speed + margins)`, so the full record
   always spans the page. At 50 mm/s a 10 s record becomes a 521 mm page:
   the same beats appear in the same pixel positions with a finer grid and
   smaller pixel amplitude (`docs/assets/gallery/speed-*.jpg`). Real devices
   keep the paper width and show 5 s at 50 mm/s or 20 s at 12.5 mm/s, with
   wider complexes. Calibration labels stay correct, but the image
   distribution for non-25 mm/s pages is unlike real pages.
2. **Stroke and grid widths are in output pixels, not millimeters.**
   `line_width` samples 0.7 to 1.6 px and `grid_width` 0.2 to 0.65 px; the
   scenarios fix 1.1 px and 0.28 px. Physical pen width therefore changes with
   output resolution, record duration and paper speed. Sub-pixel grid widths
   make 1 mm and 5 mm lines nearly indistinguishable after antialiasing
   (crops of `supervision-rgb.png`), while real paper prints 5 mm lines
   clearly heavier.
3. **Steep strokes are thinner than flat ones.** Coverage is computed per
   column, so QRS strokes print about one third as thick as baselines
   (`research/stroke-width/probe.py`; visible in the trace crops).
4. **Text is a fixed vocabulary.** The font atlas holds about forty whole
   tokens: lead names, the speed/gain line, "ECG RECORD", "RECORD SYNTHETIC"
   and five artifact words. Templates add two to four constant header lines.
   Real pages carry patient and device blocks, dates, measured intervals
   (HR, PR, QRS, QT/QTc, axes), interpretation statements, filter settings
   and footers in many layouts and fonts. A model can learn that text is rare
   and always identical.
5. **2.5D paper in the torch path barely deforms content.** In
   `round3-gallery.jpg` the crumpled and folded recipes show soft light and
   dark streaks while grid lines and traces stay straight. Real crumples
   displace lines, add sharp specular creases and self-shadowing.

### Medium

6. **No lead-switch marks or segment breaks.** In 3x4 layouts the trace runs
   continuously from one lead into the next. Most devices print a short
   vertical marker or a visible discontinuity at each column boundary.
7. **Damage looks composited.** Thermal damage is a gray smoky cloud
   (`round3-gallery.jpg`), cuts are drawn lines without a gap or lifted edge,
   and holes have clean cutouts showing the table.
8. **Missing capture modes from real datasets.** No grayscale or bilevel
   (black-and-white) scan observation exists; `bw`/`wb` are targets only.
   Scanner banding is a pure sinusoid with at most 1.8 % amplitude.
9. **Thin camera model.** Specular gloss tops out at 0.13 and scenarios use
   0.01 to 0.025, so thermal-paper glare and flash hotspots are rare. There is
   no chromatic aberration, Bayer demosaicing or phone sharpening halo, and
   JPEG is a DCT simulation rather than a codec.
10. **Display PWM banding is exaggerated.** `display_pwm` produces fully black
    bands with hard edges (`realism-v090-gallery.jpg`); real banding is a
    partial, gradual modulation.
11. **Backgrounds are few and synthetic.** Eight procedural families plus one
    CC0 wood photo. Older galleries show regular wavy grain on wood.

### Low

12. **Missing real-world artifacts listed by ECG-Image-Database:** mold,
    scanner-jam crumpling, scorch from ironing, handwriting with words,
    stickers or staples, fingers holding the page and overlapping sheets.
13. **Demo waveforms are strictly periodic** (one beat repeated every 0.85 s),
    so every gallery looks mechanical. This affects presentation only;
    training should use real recordings. 0.13.0 adds optional variability to
    `demo_signals`.
14. **No quantitative realism evidence.** All realism statements, including
    this one, are visual. There is no distribution metric (for example KID on
    patches) against ECG-Image-Database captures.

## Suggested order

1. Fix paper geometry: fixed physical page width, time window derived from
   speed, widths in millimeters (items 1 and 2). Both change outputs, so add
   them as opt-in modes first, as with the isotropic stroke (item 3).
2. Generate text from templates with random fictional fields and measurement
   blocks (item 4) and add lead-switch marks (item 6).
3. Measure before tuning visuals further: a patch-level KID and a downstream
   transfer test on ECG-Image-Database, keeping patient groups disjoint
   (item 14). Use it to rank items 5 and 7 to 11.
