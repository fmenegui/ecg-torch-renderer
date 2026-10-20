"""Physical document conventions. Templates contain fictional metadata only."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DocumentTemplate:
    """A generic report style, not a reproduction of a certified device.

    Margins and typography use millimeters in the canonical material frame.
    ``header_lines`` must fit in ``top_mm``. Speed/gain text remains generated
    from the actual rendering parameters, never from this descriptive header.
    """

    name: str
    left_mm: float = 14.0
    right_mm: float = 7.0
    top_mm: float = 22.0
    bottom_mm: float = 6.0
    font_mm: float = 2.1
    header_lines: tuple[str, ...] = (
        "SYNTHETIC ECG / RESEARCH",
        "SUBJECT: SIMULATED / ID: DEMO",
    )
    grid_rgb: tuple[float, float, float] = (0.86, 0.40, 0.45)
    paper_rgb: tuple[float, float, float] = (0.995, 0.990, 0.975)
    ink_rgb: tuple[float, float, float] = (0.035, 0.035, 0.045)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Template name must be nonempty")
        for key in ("left_mm", "right_mm", "top_mm", "bottom_mm", "font_mm"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Positive finite {key} required")
        if (
            not isinstance(self.header_lines, tuple)
            or not 1 <= len(self.header_lines) <= 4
        ):
            raise ValueError("Use a tuple of one to four header lines")
        if any(
            not isinstance(t, str) or not t or len(t) > 80 for t in self.header_lines
        ):
            raise ValueError("Header lines must contain 1..80 characters")
        if self.top_mm < 5 + (len(self.header_lines) + 1) * self.font_mm * 1.5:
            raise ValueError("Top margin cannot contain the requested typography")
        for key in ("grid_rgb", "paper_rgb", "ink_rgb"):
            value = getattr(self, key)
            if len(value) != 3 or any(
                isinstance(x, bool) or not math.isfinite(x) or not 0 <= x <= 1
                for x in value
            ):
                raise ValueError(f"{key} must be finite RGB in [0,1]")


DOCUMENT_TEMPLATES = {
    "clinical": DocumentTemplate("clinical"),
    "archive": DocumentTemplate(
        "archive",
        top_mm=26.0,
        font_mm=2.0,
        header_lines=(
            "ECG ARCHIVE / SYNTHETIC",
            "SUBJECT: SIMULATED / RECORD: 000001",
            "RESEARCH COPY / NO CLINICAL INTERPRETATION",
        ),
        grid_rgb=(0.58, 0.68, 0.77),
        paper_rgb=(0.99, 0.98, 0.94),
    ),
    "compact": DocumentTemplate(
        "compact",
        left_mm=12.0,
        right_mm=6.0,
        top_mm=19.0,
        font_mm=1.8,
        header_lines=("SIMULATED ECG / DEMONSTRATION",),
        grid_rgb=(0.75, 0.43, 0.48),
    ),
}
