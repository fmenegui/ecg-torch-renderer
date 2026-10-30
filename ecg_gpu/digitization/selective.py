"""Reference-free selection between complete ECG reconstructions.

The direct result is always retained. A candidate must preserve its finite
support, physical axes and lead identity, and agree on the shared signal.
These are engineering rejection rules, not an accuracy or clinical certificate.
No reference waveform, printed gold mask or diagnosis is accepted by this API.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Callable, Protocol

import numpy as np

from .signals import LEADS


@dataclass(frozen=True)
class SignalProposal:
    values_mv: np.ndarray
    fs: float
    source_id: str
    layout_id: str | None
    origin_s: float = 0.0
    lead_names: tuple[str, ...] = LEADS
    unit: str = "mV"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        values = np.array(self.values_mv, dtype=np.float64, copy=True)
        if values.ndim != 2 or values.shape[0] != 12 or not values.shape[1]:
            raise ValueError("A canonical [12, time] array is required")
        if np.isinf(values).any() or self.unit != "mV" or tuple(self.lead_names) != LEADS:
            raise ValueError("Explicit mV, canonical lead order and no infinity required")
        if isinstance(self.fs, bool) or not math.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("A positive finite sample rate is required")
        if not math.isfinite(self.origin_s) or self.origin_s < 0 or not self.source_id:
            raise ValueError("A physical origin and original-image identity are required")
        values.setflags(write=False)
        object.__setattr__(self, "values_mv", values)


@dataclass(frozen=True)
class GeometryEvidence:
    original_source_id: str
    original_rgb_sampled: bool
    full_frame_preserved: bool
    boundary_simple: bool
    triangle_orientation_consistent: bool
    extrapolated_fraction: float
    reference_used: bool = False

    def __post_init__(self):
        if not self.original_source_id or not math.isfinite(self.extrapolated_fraction):
            raise ValueError("A source identity and measured extrapolation are required")
        if not 0 <= self.extrapolated_fraction <= 1:
            raise ValueError("Extrapolation must be in [0,1]")


@dataclass(frozen=True)
class SpecialistEvidence:
    """A separately pinned native engine receiving the unchanged original RGB."""
    original_source_id: str
    profile: str
    assets_lock_sha256: str
    reference_used: bool = False

    def __post_init__(self):
        if len(self.assets_lock_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.assets_lock_sha256):
            raise ValueError("A verified native asset-lock digest is required")


@dataclass(frozen=True)
class SelectionPolicy:
    schema: str = "ecg-selective-policy-v2-development"
    minimum_new_samples: int = 10
    minimum_common_samples_per_lead: int = 100
    minimum_correlation: float = 0.97
    amplitude_ratio_low: float = 0.85
    amplitude_ratio_high: float = 1.18
    maximum_dc_difference_mv: float = 0.05
    maximum_timing_difference_s: float = 0.020
    maximum_extrapolated_fraction: float = 0.35

    def __post_init__(self):
        if type(self.minimum_new_samples) is not int or self.minimum_new_samples < 1:
            raise ValueError("A positive integer support gain is required")
        if type(self.minimum_common_samples_per_lead) is not int or self.minimum_common_samples_per_lead < 3:
            raise ValueError("At least three common samples are required")
        if not 0 < self.minimum_correlation <= 1:
            raise ValueError("Correlation threshold must be in (0,1]")
        if not 0 < self.amplitude_ratio_low <= 1 <= self.amplitude_ratio_high:
            raise ValueError("Amplitude-ratio bounds must surround one")
        for value in [self.maximum_dc_difference_mv, self.maximum_timing_difference_s, self.maximum_extrapolated_fraction]:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Finite bounded tolerances are required")

    @property
    def sha256(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _agreement(a, b, fs, policy):
    common = np.isfinite(a) & np.isfinite(b)
    if common.sum() < policy.minimum_common_samples_per_lead:
        return {"accepted": False, "reason": "insufficient_common_signal", "common_samples": int(common.sum())}
    x, y = a[common], b[common]
    sx, sy = float(np.std(x)), float(np.std(y))
    if min(sx, sy) < 1e-8:
        return {"accepted": False, "reason": "constant_or_nearly_constant_signal"}
    correlation = float(np.corrcoef(x, y)[0, 1])
    amplitude_ratio = sy / sx
    dc_difference = float(abs(y.mean() - x.mean()))
    radius = max(1, round(0.02 * fs))
    # Compare on the physical axis without compressing gaps or circular wrap.
    best = (-math.inf, 0)
    for lag in sorted(range(-radius, radius + 1), key=lambda value: (abs(value), value)):
        left, right = (a[lag:], b[:-lag]) if lag > 0 else (a[:lag], b[-lag:]) if lag < 0 else (a, b)
        keep = np.isfinite(left) & np.isfinite(right)
        if keep.sum() < policy.minimum_common_samples_per_lead:
            continue
        xx, yy = left[keep], right[keep]
        if min(np.std(xx), np.std(yy)) < 1e-8:
            continue
        cc = float(np.corrcoef(xx, yy)[0, 1])
        if cc > best[0]:
            best = (cc, lag)
    accepted = (best[0] >= policy.minimum_correlation
                and policy.amplitude_ratio_low <= amplitude_ratio <= policy.amplitude_ratio_high
                and dc_difference <= policy.maximum_dc_difference_mv
                and abs(best[1]) / fs <= policy.maximum_timing_difference_s)
    return {"accepted": bool(accepted), "common_samples": int(common.sum()),
            "zero_lag_correlation": correlation, "best_relative_correlation": best[0], "amplitude_std_ratio": amplitude_ratio,
            "absolute_dc_difference_mv": dc_difference, "best_relative_lag_samples": best[1]}


def select_proposal(direct, candidate, evidence, policy=None):
    """Choose a whole result, never fuse samples, fit amplitude or fill gaps."""
    policy = policy or SelectionPolicy()
    report = {"schema": "ecg-selective-decision-v1", "policy_sha256": policy.sha256,
              "selected": "direct", "reason": None, "reference_used": False,
              "numeric_support_is_not_accuracy": True, "lead_checks": {}}

    def reject(reason):
        report["reason"] = reason
        return direct, report

    if direct.source_id != candidate.source_id or evidence.original_source_id != direct.source_id:
        return reject("different_original_images")
    if direct.values_mv.shape != candidate.values_mv.shape or direct.fs != candidate.fs or direct.origin_s != candidate.origin_s:
        return reject("different_physical_time_axes")
    if not direct.layout_id or str(direct.layout_id).strip().lower() in ["unknown", "unknown layout", "none"] or direct.layout_id != candidate.layout_id:
        return reject("layout_identity_not_preserved")
    if direct.metadata.get("layout_fallback") or candidate.metadata.get("layout_fallback"):
        return reject("layout_fallback_is_not_semantic_evidence")
    if evidence.reference_used:
        return reject("reference_assisted_geometry")
    if isinstance(evidence, SpecialistEvidence):
        report.update(route="native_specialist", specialist_profile=evidence.profile,
                      specialist_assets_lock_sha256=evidence.assets_lock_sha256)
        if evidence.profile != "kaggle-second-final-six" or direct.layout_id != "3x4+1R":
            return reject("unsupported_specialist_profile_or_layout")
        length = direct.values_mv.shape[1]
        if length % 4:
            return reject("specialist_requires_four_equal_printed_segments")
        expected = np.zeros_like(direct.values_mv, dtype=bool)
        layout = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
        for row in layout:
            for column, lead in enumerate(row):
                expected[LEADS.index(lead), column * (length // 4):(column + 1) * (length // 4)] = True
        expected[LEADS.index("II")] = True
        if (np.isfinite(direct.values_mv) & ~expected).any() or (np.isfinite(candidate.values_mv) & ~expected).any():
            return reject("output_support_contradicts_specialist_layout")
    elif isinstance(evidence, GeometryEvidence):
        report["route"] = "external_rectification"
        if not all([evidence.original_rgb_sampled, evidence.full_frame_preserved,
                    evidence.boundary_simple, evidence.triangle_orientation_consistent]):
            return reject("insufficient_geometric_checks")
        if evidence.extrapolated_fraction > policy.maximum_extrapolated_fraction:
            return reject("excessive_geometric_extrapolation")
    else:
        return reject("unrecognized_candidate_evidence")
    observed, proposed = np.isfinite(direct.values_mv), np.isfinite(candidate.values_mv)
    lost, added = observed & ~proposed, proposed & ~observed
    report.update(direct_finite_samples=int(observed.sum()), candidate_finite_samples=int(proposed.sum()),
                  lost_finite_samples=int(lost.sum()), new_finite_samples=int(added.sum()))
    if lost.any():
        return reject("direct_samples_would_be_lost")
    if added.sum() < policy.minimum_new_samples:
        return reject("no_material_support_gain")
    for index, lead in enumerate(LEADS):
        if not proposed[index].any():
            continue
        check = _agreement(direct.values_mv[index], candidate.values_mv[index], direct.fs, policy)
        report["lead_checks"][lead] = check
        if not check["accepted"]:
            return reject("shared_signal_disagreement:" + lead)
    report.update(selected="candidate", reason="support_gain_with_preserved_shared_signal")
    return candidate, report


class ImageBackend(Protocol):
    def predict(self, rgb, *, fs, samples, source_id) -> SignalProposal:
        ...


class SelectiveDigitizer:
    """Run a direct backend and an optional image-only rectification branch."""
    def __init__(self, backend: ImageBackend, rectifier: Callable | None = None, policy=None, specialist=None):
        self.backend, self.rectifier = backend, rectifier
        self.policy = policy or SelectionPolicy()
        self.specialist = specialist

    def predict(self, rgb, *, fs, samples):
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("A nonempty uint8 RGB image is required")
        original = np.array(image, copy=True)
        original.setflags(write=False)
        source_id = hashlib.sha256(str(original.shape).encode() + original.tobytes()).hexdigest()
        direct = self.backend.predict(original.copy(), fs=fs, samples=samples, source_id=source_id)
        if direct.source_id != source_id or direct.fs != fs or direct.values_mv.shape != (12, samples):
            raise ValueError("Direct backend changed the requested identity or time axis")
        report = {"selected": "direct", "reason": "no_rectifier", "policy_sha256": self.policy.sha256,
                  "reference_used": False, "source_id": source_id}
        specialist_decision = None
        if self.specialist is not None and direct.layout_id == "3x4+1R":
            try:
                proposal, evidence = self.specialist(original.copy(), fs=fs, samples=samples, source_id=source_id)
                selected, specialist_decision = select_proposal(direct, proposal, evidence, self.policy)
                if selected is proposal:
                    specialist_decision["source_id"] = source_id
                    return selected, specialist_decision
            except Exception as error:
                specialist_decision = {"selected": "direct", "reason": "specialist_branch_failed", "error_type": type(error).__name__, "error": str(error)}
            report["specialist_decision"] = specialist_decision
        if self.rectifier is None:
            return direct, report
        try:
            corrected, evidence = self.rectifier(original.copy(), source_id=source_id)
            if np.array_equal(corrected, original):
                report["reason"] = "unchanged_rgb"
                return direct, report
            candidate = self.backend.predict(corrected, fs=fs, samples=samples, source_id=source_id)
            selected, report = select_proposal(direct, candidate, evidence, self.policy)
            report["source_id"] = source_id
            if specialist_decision is not None:
                report["specialist_decision"] = specialist_decision
            return selected, report
        except Exception as error:
            report.update(reason="candidate_branch_failed", error_type=type(error).__name__, error=str(error))
            return direct, report
