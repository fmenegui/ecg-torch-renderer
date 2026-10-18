"""Aligned image/segmentation/detection batches from one ECG render.

All outputs remain on the input device. The renderer is a non-differentiable
data transform. No image encoding or CPU transfer occurs during forward.
"""
from dataclasses import asdict, dataclass
import math

import torch

from .gpu_module import ECGImageLayer
from .schema_v05 import LEADS


_OUTPUTS = frozenset({
    'lead_masks', 'trace_mask', 'bw', 'wb', 'lead_boxes',
    'instance_masks', 'instance_boxes',
})


@dataclass(frozen=True)
class SupervisionConfig:
    """Opt-in supervision. RGB and visibility are always returned.

    ``leads`` sets channel order, including optional subsets. ``binary=False``
    preserves anti-aliased coverage. Binary masks and boxes use ``> threshold``.
    ``visible_only`` removes torn, cropped and explicitly occluded pixels.
    Boxes are xyxy with exclusive right/bottom edges. Empty boxes are -1.

    Lead masks group *all occurrences* of a canonical lead. Lead boxes are
    their joint envelope, so detection of individual short/rhythm strips must
    use instance boxes instead. Instance outputs retain all 15 printed slots,
    regardless of ``leads``. Their canonical IDs and active flags identify
    repetitions and padded slots without changing the class vocabulary.
    """
    outputs: tuple[str, ...] = ('lead_masks',)
    leads: tuple[str, ...] = tuple(LEADS)
    binary: bool = False
    threshold: float = .05
    visible_only: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'outputs', tuple(self.outputs))
        object.__setattr__(self, 'leads', tuple(self.leads))
        if len(set(self.outputs)) != len(self.outputs) or set(self.outputs) - _OUTPUTS:
            raise ValueError(f'Unique supervision outputs required, choose from {sorted(_OUTPUTS)}')
        if not self.leads or len(set(self.leads)) != len(self.leads) or set(self.leads) - set(LEADS):
            raise ValueError(f'Unique standard lead names required, choose from {list(LEADS)}')
        if not math.isfinite(self.threshold) or not 0 <= self.threshold < 1:
            raise ValueError('threshold must be finite and in [0,1)')
        if not isinstance(self.binary, bool) or not isinstance(self.visible_only, bool):
            raise ValueError('binary and visible_only must be bool')


def mask_bounding_boxes(masks, threshold=.05):
    """Return GPU-safe ``(boxes[B,C,4], present[B,C])`` from BCHW masks.

    A box encloses pixels whose coverage exceeds threshold. Coordinates use
    integer pixel edges [x0,y0,x1,y1], with x1/y1 exclusive. Empty slots use -1.
    No host reads, variable-length lists, or per-lead Python loops are used.
    """
    if masks.ndim != 4:
        raise ValueError('masks must have shape [B,C,H,W]')
    if not math.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError('threshold must be finite and in [0,1)')
    h, w = masks.shape[-2:]
    if not h or not w:
        raise ValueError('masks must have nonempty image dimensions')
    occupied = masks > threshold
    columns = occupied.any(-2)
    rows = occupied.any(-1)
    xs = torch.arange(w, device=masks.device)[None, None]
    ys = torch.arange(h, device=masks.device)[None, None]
    x0 = torch.where(columns, xs, w).amin(-1)
    x1 = torch.where(columns, xs + 1, 0).amax(-1)
    y0 = torch.where(rows, ys, h).amin(-1)
    y1 = torch.where(rows, ys + 1, 0).amax(-1)
    present = (x1 > x0) & (y1 > y0)
    boxes = torch.stack((x0, y0, x1, y1), -1)
    return torch.where(present[..., None], boxes, -1), present


class ECGSupervisionLayer(ECGImageLayer):
    """Signal-to-batch nn.Module for segmentation and detection training.

    ``batch = layer(signals)`` renders exactly once. ``batch['image']`` is
    [B,3,H,W], ``batch['valid']`` is [B,1,H,W]. Requested masks are float32.
    Canonical lead identity is retained even for repeated rhythm strips and
    Cabrera's inverted aVR. Use independent sigmoid/BCE channels at crossings.
    Masks describe waveform ink, excluding lead-name text, calibration pulses
    and grid. Boxes tightly enclose that ink rather than an allocated cell.
    Instance ``active`` means enabled before capture, while ``box_valid``
    means target pixels survived visibility and the coverage threshold.
    The default returns 12 soft lead masks. Image-only networks can use
    ECGImageLayer instead to avoid all dense supervision allocations.
    """
    def __init__(self, fs=500., *, supervision=None, photo=True, **options):
        super().__init__(fs=fs, **options)
        if supervision is not None and not isinstance(supervision, SupervisionConfig):
            raise TypeError('supervision must be a SupervisionConfig')
        self.supervision = supervision or SupervisionConfig()
        if not isinstance(photo, bool):
            raise ValueError('photo must be bool')
        self.photo = photo
        indices = [LEADS.index(lead) for lead in self.supervision.leads]
        self.register_buffer('_lead_indices', torch.tensor(indices, device=self.renderer.device), persistent=False)

    def get_extra_state(self):
        return {**super().get_extra_state(), 'supervision': asdict(self.supervision), 'photo': self.photo}

    def set_extra_state(self, state):
        if state.get('supervision') != asdict(self.supervision) or state.get('photo') != self.photo:
            raise ValueError('Supervision configuration changed. Recreate the same layer before resuming')
        super().set_extra_state(state)

    @torch.no_grad()
    def forward(self, signals, fs=None, *, lengths=None, parameters=None, generator=None):
        requested = self.supervision.outputs
        instances_needed = bool({'instance_masks', 'instance_boxes'} & set(requested))
        rendered = self.render_batch(
            signals, fs, lengths=lengths, parameters=parameters, generator=generator,
            stage='trace', photo=self.photo, return_instances=instances_needed,
            return_baselines=False, return_instance_boxes=False,
        )
        return self.supervise(rendered)

    @torch.no_grad()
    def supervise(self, rendered):
        """Convert an existing paired RenderBatch without rendering again.

        Useful when a caller also needs scene maps or full provenance. Render
        with stage='trace' or 'all', and return_instances=True when requested.
        """
        requested = self.supervision.outputs
        instances_needed = bool({'instance_masks', 'instance_boxes'} & set(requested))
        required = {'valid', 'traces'} | ({'instances'} if instances_needed else set())
        if required - rendered.targets.keys():
            raise ValueError(f'RenderBatch lacks supervision targets: {sorted(required - rendered.targets.keys())}')
        valid = rendered.targets['valid']
        batch = {'image': rendered.image, 'valid': valid}
        lead_outputs = {'lead_masks', 'lead_boxes'} & set(requested)
        union_outputs = {'trace_mask', 'bw', 'wb'} & set(requested)
        traces = None
        if lead_outputs or union_outputs:
            traces = rendered.targets['traces']
            if self.supervision.leads != tuple(LEADS):
                traces = traces.index_select(1, self._lead_indices)
            # Coverage and visibility are nonnegative. Max reduction commutes
            # with visibility and thresholding, so union-only targets need no
            # additional full per-lead tensor. Instance-only requests bypass
            # lead processing altogether. Source masks remain unmodified.
            if not lead_outputs:
                traces = traces.amax(1, keepdim=True)
            if self.supervision.visible_only:
                traces = traces * valid
            if self.supervision.binary:
                traces = (traces > self.supervision.threshold).to(traces.dtype)
        if 'lead_masks' in requested:
            batch['lead_masks'] = traces
            batch['lead_ids'] = self._lead_indices[None].expand(rendered.image.shape[0], -1)
        union = None
        if union_outputs:
            union = traces.amax(1, keepdim=True) if lead_outputs else traces
        if 'trace_mask' in requested:
            batch['trace_mask'] = union
        if 'bw' in requested:
            batch['bw'] = 1 - union
        if 'wb' in requested:
            batch['wb'] = union
        if 'lead_boxes' in requested:
            batch['lead_boxes'], batch['lead_box_valid'] = mask_bounding_boxes(traces, self.supervision.threshold)
            batch['lead_ids'] = self._lead_indices[None].expand(rendered.image.shape[0], -1)
        if instances_needed:
            instance_masks = rendered.targets['instances']
            if self.supervision.visible_only:
                instance_masks = instance_masks * valid
            if self.supervision.binary:
                instance_masks = (instance_masks > self.supervision.threshold).to(instance_masks.dtype)
            if 'instance_masks' in requested:
                batch['instance_masks'] = instance_masks
            if 'instance_boxes' in requested:
                batch['instance_boxes'], batch['instance_box_valid'] = mask_bounding_boxes(instance_masks, self.supervision.threshold)
            info = rendered.metadata
            batch['instance_lead_ids'] = info['lead_id']
            batch['instance_active'] = info['active']
            batch['instance_windows_s'] = torch.stack((info['start_s'], info['end_s']), -1)
            batch['instance_amplitude_sign'] = info['amplitude_sign']
        return batch

