"""Experimental original-RGB grid correction as a selective candidate backend."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .adaptive_sampling import render_mesh
from .complete_rectifier import fit_complete_map
from .denoising_grid import DenoisingGridNet
from .lattice import organize_lattice
from .local_cells import locally_supported_cells
from .native_detection import detect_native_grid
from .observation import remove_periodic_banding
from .page_orientation import infer_text_orientation, rotate_lattice_ids
from .selective import GeometryEvidence

SELECTED_GRID_SHA256 = "8115e99c3517dc0b99974b9e64744aab9eb961fd0da06f1e84a65a1e45c3dbf8"


class AdaptiveGridRectifier:
    """Fit from learned grid observations, then sample the original RGB only.

    Default weights are the previously validation-selected joint-1973 model.
    Passing another hash is an explicitly changed experimental checkpoint.
    The denoising RGB head is never used as a reconstructed ECG image.
    """
    def __init__(self, checkpoint, *, device="cpu", max_side=3000,
                 expected_sha256=SELECTED_GRID_SHA256, text_orientation=True):
        self.checkpoint = Path(checkpoint)
        if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Selected grid checkpoint identity mismatch")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for grid inference, unavailable")
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if state.get("model_class") != "DenoisingGridNet":
            raise ValueError("A declared DenoisingGridNet checkpoint is required")
        self.model = DenoisingGridNet(width=state["width"]).to(device).eval()
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.device, self.max_side = device, max_side
        self.threshold, self.text_orientation = float(state["threshold"]), text_orientation
        self.checkpoint_sha256 = expected_sha256
        self.last_report = None

    def __call__(self, rgb, *, source_id):
        import shapely
        self.last_report = None
        image = np.asarray(rgb)
        xy, scores, detection = detect_native_grid(self.model, image, device=self.device,
            threshold=self.threshold, global_size=(704, 1024), max_tiles=0, max_points=16384)
        filtered, banding = remove_periodic_banding(image, xy)
        if banding["applied"]:
            xy, scores, detection = detect_native_grid(self.model, filtered, device=self.device,
                threshold=self.threshold, global_size=(704, 1024), max_tiles=0, max_points=16384)
        native = organize_lattice(xy, scores)
        local = locally_supported_cells(xy, native)
        orientation = infer_text_orientation(image) if self.text_orientation else {"quarter_turns_ccw": 0, "status": "disabled"}
        local = rotate_lattice_ids(local, orientation["quarter_turns_ccw"])
        ids = local.get("ids")
        chosen = np.asarray(local.get("selected_indices", []), int)
        observed = xy[chosen]
        if ids is None or len(observed) < 16:
            raise ValueError("Insufficient supported grid correspondences")
        ids, observed = np.asarray(ids), np.asarray(observed)
        mapping, fit = fit_complete_map(observed, ids[:, ::-1] * 5.0, ids, image.shape,
            scores=scores[chosen], cells=local["cells"], global_prior="projective")
        if mapping is None:
            raise ValueError("Grid completion rejected: " + str(fit.get("reason")))
        corrected, _, sampling = render_mesh(image, mapping.source, mapping.target, mapping.triangles, max_side=self.max_side)
        polygon = shapely.union_all(shapely.polygons(mapping.target[mapping.triangles]))
        known = mapping.forward(observed)
        hull = shapely.convex_hull(shapely.multipoints(known[np.isfinite(known).all(1)]))
        outside = float(np.clip(1 - polygon.intersection(hull).area / polygon.area, 0, 1))
        self.last_report = {"schema": "adaptive-grid-candidate-v1", "checkpoint_sha256": self.checkpoint_sha256,
            "assumed_major_grid_mm": 5.0, "detector": detection, "banding": banding,
            "orientation": orientation, "fit": fit, "sampling": sampling,
            "outside_observed_hull_fraction": outside, "outside_hull_denominator": "continuous target mesh area",
            "reference_used": False, "learned_rgb_restoration_used": False,
            "geometry_accuracy_verified": False}
        evidence = GeometryEvidence(source_id, True, True, True, True, outside)
        return corrected, evidence
