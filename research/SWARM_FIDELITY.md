# ECG document fidelity audit

Checked on 2026-10-18 with Python 3.12 and PyTorch 2.14.1+cpu. No CUDA device was available. The fidelity tests set PyTorch to two CPU threads.

## Confirmed defect fixed

`LayoutSpec` was a frozen dataclass but retained mutable nested lists passed by a caller. A later edit could change the layout reported by the manifest while the renderer continued to use tables built from its earlier contents. Construction now snapshots rows and rhythm leads into nested tuples. Malformed names, row structures and non-string lead tokens raise `ValueError` at construction.

Normal tuple-based layouts and their rendering remain unchanged. Callers can still supply nested lists, which are copied. The raster kernels were deliberately preserved because the repository maintains frozen v0.2.1 comparisons.

## Verification

Command: `python -m unittest discover -s tests -p test_fidelity.py -v`

Result: **8 tests passed in 0.781 seconds**.

The tests cover:

- Every built-in layout's lead identity, row, column, amplitude polarity and sequential time window.
- The Cabrera sequence `aVL, I, -aVR, II, aVF, III, V1...V6`. An isolated positive aVR input is drawn below baseline in the inverted display, while its target remains in the original aVR channel and the source remains unchanged.
- Simultaneous short-lead windows starting at zero, full-duration rhythm strips and per-sample rhythm lead overrides.
- All nine speed/gain combinations from 12.5, 25 and 50 mm/s and 5, 10 and 20 mm/mV. The same pixels-per-mm scalar governs the two paper axes.
- An independent raster measurement of the 1 mV, 0.2 second calibration mark for every speed/gain pair. At 25 mm/s and 10 mm/mV this gives a 5 mm width and 10 mm height. At 50 mm/s its width is 10 mm. Absent calibration marks produce zero coverage.
- Linear interpolation with different valid source lengths, including queries beyond the valid end. Padding sentinels cannot enter the interpolated signal.
- Immutable layout snapshots and malformed-layout rejection.

The calibration reference is the FIU Essential Electrocardiography teaching resource: https://ilearn.courses.med.fiu.edu/published/Essential_ECG/ECG_intro/std.html

## Interpretation and limits

The renderer describes duration as the time between the first and last retained sample, `(N - 1) / fs`, capped by `max_duration_s`. This is internally consistent with its interpolation and metadata. It differs by one sample interval from the nominal acquisition-window convention `N / fs`.

Rows fit the configured image height. Large voltages or gains can cross adjacent row bands or leave the page. The renderer does not silently clip the physiological signal to a row band. Low-resolution outputs can alias narrow ECG features despite the original-sample envelope preservation.

These checks establish physical scaling and layout semantics. They do not establish that input signals satisfy limb-lead identities or represent a physiologically possible ECG. They do not constitute clinical validation, scanner calibration, real-photo equivalence or GPU performance validation. Procedural paper geometry remains a bounded height-field approximation. Manufacturer-specific report templates and a blinded comparison against real ECG images remain separate validation work.
