"""Aligned image/segmentation/detection batches from one ECG render.

All outputs remain on the input device. The renderer is a non-differentiable
data transform. No image encoding or CPU transfer occurs during forward.
"""
from dataclasses import asdict, dataclass
import math

import torch

from .gpu_module import ECGImageLayer
from .gpu_renderer import _conditional_visibility
from .schema_v05 import LEADS
from .grid_points import GridPointConfig


_OUTPUTS = frozenset({
    'lead_masks', 'trace_mask', 'bw', 'wb', 'lead_boxes',
    'instance_masks', 'instance_boxes',
    'instance_visible_fraction',
    'name_masks', 'name_boxes',
    'grid_points_mask', 'grid_points',
})


@dataclass(frozen=True)
class SupervisionConfig:
    """Opt-in supervision. RGB and visibility are always returned.

    ``leads`` sets channel order, including optional subsets. ``binary=False``
    preserves anti-aliased coverage. Binary masks and boxes use ``> threshold``.
    ``visible_only`` removes torn, cropped and explicitly occluded pixels.
    Fractional sheet-edge coverage is applied once, including after warping.
    Boxes are xyxy with exclusive right/bottom edges. Empty boxes are -1.

    Lead masks group *all occurrences* of a canonical lead. Lead boxes are
    their joint envelope, so detection of individual short/rhythm strips must
    use instance boxes instead. Instance outputs retain all 15 printed slots,
    regardless of ``leads``. Their canonical IDs and active flags identify
    repetitions and padded slots without changing the class vocabulary.
    ``name_masks`` and ``name_boxes`` independently describe the lead-name
    glyphs for those same printed slots, including repeated rhythm labels.
    """
    outputs: tuple[str, ...] = ('lead_masks',)
    leads: tuple[str, ...] = tuple(LEADS)
    binary: bool = False
    threshold: float = .05
    visible_only: bool = True
    grid_point_config: GridPointConfig = GridPointConfig()

    def __post_init__(self):
        object.__setattr__(self, 'outputs', (self.outputs,) if isinstance(self.outputs, str) else tuple(self.outputs))
        object.__setattr__(self, 'leads', (self.leads,) if isinstance(self.leads, str) else tuple(self.leads))
        if len(set(self.outputs)) != len(self.outputs) or set(self.outputs) - _OUTPUTS:
            raise ValueError(f'Unique supervision outputs required, choose from {sorted(_OUTPUTS)}')
        if not self.leads or len(set(self.leads)) != len(self.leads) or set(self.leads) - set(LEADS):
            raise ValueError(f'Unique standard lead names required, choose from {list(LEADS)}')
        if not math.isfinite(self.threshold) or not 0 <= self.threshold < 1:
            raise ValueError('threshold must be finite and in [0,1)')
        if not isinstance(self.binary, bool) or not isinstance(self.visible_only, bool):
            raise ValueError('binary and visible_only must be bool')
        if not isinstance(self.grid_point_config, GridPointConfig):
            raise TypeError('grid_point_config must be GridPointConfig')
        if not self.visible_only and {'grid_points', 'grid_points_mask'} & set(self.outputs):
            raise ValueError('Grid points require visible_only=True. Hidden points are not reconstructed')
        if self.grid_point_config.return_offsets and 'grid_points_mask' in self.outputs and 'grid_points' not in self.outputs:
            raise ValueError("Include 'grid_points' in outputs to return the requested offset targets")


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
    Waveform masks exclude lead-name text, calibration pulses and grid.
    Optional name masks/boxes instead enclose the lead-name glyph support.
    Boxes tightly enclose the requested ink rather than an allocated cell.
    Instance ``active`` means enabled before capture, while ``box_valid``
    means target pixels survived visibility and the coverage threshold.
    The default returns 12 soft lead masks. Image-only networks can use
    ECGImageLayer instead to avoid all dense supervision allocations.
    """
    def __init__(self, fs=500., *, supervision=None, photo=True, **options):
        if supervision is not None and not isinstance(supervision, SupervisionConfig):
            raise TypeError('supervision must be a SupervisionConfig')
        if not isinstance(photo, bool):
            raise ValueError('photo must be bool')
        resolved = supervision if supervision is not None else SupervisionConfig()
        super().__init__(fs=fs, photo=photo, **options)
        self.supervision = resolved
        self.photo = photo
        indices = [LEADS.index(lead) for lead in self.supervision.leads]
        self.register_buffer('_lead_indices', torch.tensor(indices, device=self.renderer.device), persistent=False)

    @staticmethod
    def _effective_supervision(configuration):
        if not isinstance(configuration, dict):
            return configuration
        configuration = dict(configuration)
        if not {'grid_points', 'grid_points_mask'} & set(configuration.get('outputs', ())):
            configuration.pop('grid_point_config', None)
        return configuration

    def get_extra_state(self):
        return {**super().get_extra_state(),
                'supervision': self._effective_supervision(asdict(self.supervision)), 'photo': self.photo}

    def set_extra_state(self, state):
        if (self._effective_supervision(state.get('supervision')) != self._effective_supervision(asdict(self.supervision))
                or state.get('photo') != self.photo):
            raise ValueError('Supervision configuration changed. Recreate the same layer before resuming')
        super().set_extra_state(state)

    def render_batch(self, signals, fs=None, *, lengths=None, parameters=None,
                     generator=None, stage='all', photo=None, grid_point_config=None, **options):
        """Render directly, inheriting this layer's grid settings by default.

        Explicit overrides remain available for inspection, but supervise()
        rejects grid targets generated with a different configuration.
        """
        if options.get('return_grid_points', False) and grid_point_config is None:
            grid_point_config = self.supervision.grid_point_config
        return super().render_batch(signals, fs, lengths=lengths, parameters=parameters,
            generator=generator, stage=stage, photo=photo, grid_point_config=grid_point_config, **options)

    @torch.no_grad()
    def forward(self, signals, fs=None, *, lengths=None, parameters=None, generator=None):
        requested = self.supervision.outputs
        instances_needed = bool({'instance_masks', 'instance_boxes', 'instance_visible_fraction'} & set(requested))
        names_needed = bool({'name_masks', 'name_boxes'} & set(requested))
        points_only = bool(requested) and set(requested) <= {'grid_points', 'grid_points_mask'}
        rendered = self.render_batch(
            signals, fs, lengths=lengths, parameters=parameters, generator=generator,
            stage='image' if points_only else 'trace', photo=self.photo, return_instances=instances_needed,
            return_baselines=False, return_instance_boxes=False,
            return_name_instances=names_needed,
            return_grid_points=bool({'grid_points', 'grid_points_mask'} & set(requested)),
            grid_point_config=self.supervision.grid_point_config,
        )
        return self.supervise(rendered)

    @torch.no_grad()
    def supervise(self, rendered):
        """Convert an existing paired RenderBatch without rendering again.

        Useful when a caller also needs scene maps or full provenance. Render
        with stage='trace' or 'all', and return_instances=True when requested.
        """
        requested = self.supervision.outputs
        instances_needed = bool({'instance_masks', 'instance_boxes', 'instance_visible_fraction'} & set(requested))
        names_needed = bool({'name_masks', 'name_boxes'} & set(requested))
        traces_needed = bool({'lead_masks', 'lead_boxes', 'trace_mask', 'bw', 'wb'} & set(requested))
        required = {'valid'} | ({'traces'} if traces_needed else set()) | ({'instances'} if instances_needed else set())
        if names_needed:
            required.add('name_instances')
        if required - rendered.targets.keys():
            raise ValueError(f'RenderBatch lacks supervision targets: {sorted(required - rendered.targets.keys())}')
        valid = rendered.targets['valid']
        # Warped masks already contain fractional sheet silhouette coverage.
        # Apply only conditional visibility here. Keep full valid for losses.
        visibility = _conditional_visibility(valid, rendered.targets.get('frame_coverage'))
        batch = {'image': rendered.image, 'valid': valid}
        if names_needed:
            names = rendered.targets['name_instances']
            if self.supervision.visible_only:
                names = names * visibility
            if self.supervision.binary:
                names = (names > self.supervision.threshold).to(names.dtype)
            if 'name_masks' in requested:
                batch['name_masks'] = names
            if 'name_boxes' in requested:
                batch['name_boxes'], batch['name_box_valid'] = mask_bounding_boxes(names, self.supervision.threshold)
            batch['name_lead_ids'] = rendered.metadata['lead_id']
            batch['name_active'] = (rendered.metadata['active']
                                    & (rendered.metadata['parameters']['label_present'] > 0))
            batch['name_amplitude_sign'] = rendered.metadata['amplitude_sign']
        if {'grid_points', 'grid_points_mask'} & set(requested):
            if 'grid_points_mask' not in rendered.targets:
                raise ValueError('Render with return_grid_points=True for grid point supervision')
            provenance = rendered.metadata.get('grid_points', {})
            if provenance.get('configuration') != asdict(self.supervision.grid_point_config):
                raise ValueError('Grid-point target configuration differs. Render with this layer or the same grid_point_config')
            if 'grid_points' in requested:
                required_points = {'grid_points_xy', 'grid_points_valid', 'grid_points_active',
                    'grid_points_ids', 'grid_points_canonical_xy', 'grid_points_mm',
                    'grid_points_error_px', 'grid_points_error_image_px'}
                if self.supervision.grid_point_config.return_offsets:
                    required_points |= {'grid_points_offsets', 'grid_points_offset_valid',
                                        'grid_points_anchor_xy', 'grid_points_anchor_valid'}
                if required_points - rendered.targets.keys():
                    raise ValueError(f'RenderBatch lacks grid-point targets: {sorted(required_points - rendered.targets.keys())}')
                batch.update({name: value for name, value in rendered.targets.items() if name.startswith('grid_points_')})
            else:
                batch['grid_points_mask'] = rendered.targets['grid_points_mask']
            if self.supervision.binary:
                batch['grid_points_mask'] = (batch['grid_points_mask'] > self.supervision.threshold).float()
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
                traces = traces * visibility
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
            fraction = rendered.targets.get('instance_visible_fraction')
            if fraction is not None:
                batch['instance_visible_fraction'] = fraction
            elif 'instance_visible_fraction' in requested:
                raise ValueError('Render with return_visibility_fraction=True for instance_visible_fraction')
            instance_masks = rendered.targets['instances']
            if self.supervision.visible_only:
                instance_masks = instance_masks * visibility
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


