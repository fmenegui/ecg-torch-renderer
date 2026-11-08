"""Observe unchanged native processing and re-read traces on a physical clock.

Observation hooks return exactly the original native results. They are installed
only while the owning backend holds its inference lock and always restored.
The alternative readout requires an explicit clock origin and calibration.
"""

from copy import deepcopy

import numpy as np

from .physical_trace import sample_physical_trace
from .selective import SignalProposal
from .signals import LEADS


class NativeTraceObserver:
    def __init__(self, model):
        self.model = model
        self.arrays = {}
        self.report = {'schema': 'native-raster-trace-observations-v1', 'reference_used': False}
        self.originals = []

    def replace(self, owner, name, wrapper):
        existed = name in owner.__dict__
        original = getattr(owner, name)
        self.originals.append((owner, name, existed, original))
        setattr(owner, name, wrapper(original))

    def __enter__(self):
        def preprocess(original):
            def observed(lines):
                result = original(lines)
                raw = lines.detach().cpu().numpy()
                keep = np.any(np.isfinite(raw) & (raw != 0), axis=0)
                indices = np.flatnonzero(keep)
                self.report.update(full_aligned_width=raw.shape[1], extraction_first_x=int(indices[0]),
                                   extraction_last_x=int(indices[-1]))
                return result
            return observed

        def normalize(original):
            def observed(lines, *args, **kwargs):
                self.arrays['merged_rows_y_px'] = lines.detach().cpu().numpy().copy()
                self.arrays['baseline_y_px'] = lines.nanmean(dim=1).detach().cpu().numpy().copy()
                return original(lines, *args, **kwargs)
            return observed

        def canonicalize(original):
            def observed(lines, match):
                self.report['native_match'] = deepcopy(match)
                self.arrays['native_rows_uv'] = lines.detach().cpu().numpy().copy()
                result = original(lines, match)
                self.arrays['native_canonical_uv'] = result.detach().cpu().numpy().copy()
                return result
            return observed

        try:
            self.replace(self.model.signal_extractor, 'preprocess_lines', preprocess)
            self.replace(self.model.identifier, 'normalize', normalize)
            self.replace(self.model.identifier, '_canonicalize_lines', canonicalize)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for owner, name, existed, original in reversed(self.originals):
            if existed:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.originals.clear()


def physical_clock_proposal(native, observer, *, source_id, fs, samples,
                            origin_x_px, origin_basis, paper_speed_mm_s, gain_mm_mv):
    """Experimental clock ablation, reusing native row/lead decisions.

    Native per-row mean and amplitude density are retained to isolate the time
    readout. The caller's origin is never fitted against a waveform reference.
    An aligned-frame origin is an explicit hypothesis, not verified print zero.
    """
    from .npj_backend import canonical_layout
    report, arrays = observer.report, observer.arrays
    match = report['native_match']
    if match.get('flip') or not np.isfinite(match.get('cost', np.inf)):
        raise ValueError('A resolved native orientation and layout are required')
    if not isinstance(origin_basis, str) or not origin_basis:
        raise ValueError('Explicit image-clock origin provenance required')
    definition = native.model.identifier.layouts[match['layout']]
    raw = arrays['merged_rows_y_px']
    spacing = report['pixel_spacing_mm']
    ppmm_x = 1 / spacing['x']
    ppmm_y = spacing['average_pixel_per_mm']
    physical, receipt = sample_physical_trace(raw, np.arange(raw.shape[1]) + report['extraction_first_x'],
        baseline_y_px=arrays['baseline_y_px'], x_origin_px=origin_x_px,
        pixels_per_mm_x=ppmm_x, pixels_per_mm_y=ppmm_y,
        paper_speed_mm_s=paper_speed_mm_s, gain_mm_mv=gain_mm_mv,
        fs=fs, samples=samples, max_gap_s=1.01 / (ppmm_x * paper_speed_mm_s))
    canonical = np.full((12, samples), np.nan)
    cols = definition['layout']['cols']
    for row, lead_row in enumerate(definition['leads']):
        for col, name in enumerate(lead_row if isinstance(lead_row, list) else [lead_row]):
            sign = -1 if name.startswith('-') else 1
            name = name.lstrip('-')
            if name in LEADS and row < len(physical):
                start, end = col * (samples // cols), (col + 1) * (samples // cols) if col < cols - 1 else samples
                canonical[LEADS.index(name), start:end] = sign * physical[row, start:end]
    rhythm_assignments = []
    count = len(definition.get('rhythm_leads', []))
    for row in range(len(raw) - count, len(raw)):
        candidates = [i for i in range(12) if np.array_equal(arrays['native_rows_uv'][row], arrays['native_canonical_uv'][i], equal_nan=True)]
        if len(candidates) != 1:
            raise ValueError('Cannot preserve the exact native rhythm assignment')
        canonical[candidates[0]] = physical[row]
        rhythm_assignments.append({'row': row, 'lead': LEADS[candidates[0]]})
    receipt.update(backend='npj-physical-x-clock-ablation', source_id=source_id,
        native_layout=match['layout'], origin_basis=origin_basis,
        baseline_inferred_from_trace_mean=True, baseline_source='unchanged native per-row centering',
        origin_is_independently_validated_print_zero=False,
        native_amplitude_density_and_row_centering_retained=True,
        rhythm_assignments=rhythm_assignments, original_article_full_pipeline=False,
        native_upstream_source_modified=False, reference_used=False,
        native_extraction_crop_first_x=report['extraction_first_x'],
        full_aligned_width=report['full_aligned_width'])
    return SignalProposal(canonical, fs, source_id, canonical_layout(match['layout']), metadata=receipt)
