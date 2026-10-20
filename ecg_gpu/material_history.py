"""An explicit illustrative liquid, drying and storage history."""

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class MaterialHistory:
    """Coupled appearance/relief prior, not a fitted fluid or chemistry model.

    Wetness decays exponentially with humidity-dependent drying time. Deposits
    accumulate as liquid evaporates. Residual local cockling remains after
    drying. Storage age, UV dose and heat affect the thermal coating separately.
    All coefficients are illustrative and require real-material calibration.
    """

    spill_ml: float = 0.0
    minutes_since_spill: float = 60.0
    relative_humidity: float = 0.5
    solute_fraction: float = 0.35
    storage_days: float = 0.0
    uv_dose_kj_m2: float = 0.0
    heat_minutes_at_60c: float = 0.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Nonnegative finite {key} required")
        if self.spill_ml > 10 or max(self.relative_humidity, self.solute_fraction) > 1:
            raise ValueError("Use spill <=10mL, humidity and solute fractions in [0,1]")

    def specification(self):
        return {
            "model": "illustrative-drying-storage-history-v1",
            "config": asdict(self),
            "calibrated": False,
        }

    def parameters(self, p):
        wetness = math.exp(
            -self.minutes_since_spill / (20 * (1 + 3 * self.relative_humidity))
        )
        volume = 1 - math.exp(-self.spill_ml)
        age = 1 - math.exp(-self.storage_days / 3650)
        fade = 1 - math.exp(-self.uv_dose_kj_m2 / 500 - self.heat_minutes_at_60c / 90)
        darkening = 1 - math.exp(-self.heat_minutes_at_60c / 120)

        def full(value):
            return torch.full_like(p["paper_age"], value)

        return {
            **p,
            "paper_age": full(age),
            "foxing_strength": full(age * self.relative_humidity),
            "thermal_fade": full(fade),
            "thermal_darkening_strength": full(darkening),
            "stain_strength": full(0.0),
            "water_strength": full(volume * wetness),
            "coffee_strength": full(volume * self.solute_fraction * (1 - wetness)),
            "coffee_radius": full(0.05 + 0.12 * math.sqrt(volume)),
            "history_warp_strength": full(0.012 * volume * (0.25 + 0.75 * wetness)),
        }
