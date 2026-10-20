"""On-device ECG print/photograph renderer with paired supervision.

Input signals are [B, 12, T], mV, in schema_v05.LEADS order. Rendering does
not alter the physiological signals. Font rasterization happens once at
construction, then the atlas and every per-batch operation live in torch.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .gpu_appearance import CameraAppearance, srgb_to_linear, linear_to_srgb
from .gpu_material import PaperMaterial
from .gpu_document import DocumentAssets, DocumentScene, SURFACES, BACKGROUNDS, SHADOWS
from .gpu_layouts import LayoutSpec, standard_layouts
from .gpu_raster import raster_kernels
from .schema_v05 import LEADS, SPEEDS, GAINS, LAYOUTS


def _conditional_visibility(valid, frame_coverage=None):
    """Visibility given a sheet hit, avoiding a second silhouette alpha."""
    if frame_coverage is None:
        return valid
    return torch.where(frame_coverage > 0,
                       valid / frame_coverage.clamp_min(torch.finfo(valid.dtype).tiny),
                       0).clamp(0, 1)


def material_pixel_area(field, height, width):
    """Canonical pixel area per output pixel for a continuous inverse map.

    The inverse Jacobian removes perspective/relief magnification from ink
    integrals. This is pixel quadrature, not an optical readability estimate.
    """
    dy, dx = torch.gradient(field, dim=(1, 2))
    determinant = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    return determinant.abs()[:, None] * (height * width / 4)


@dataclass(frozen=True)
class RenderConfig:
    height: int = 768
    width: int = 1024
    profile: str = 'mixed'
    severity: float = .6
    antialias_samples: int = 4
    lead_chunk_size: int = 3
    max_duration_s: float = 10.
    jpeg_probability: float = .65
    preserve_extrema: bool = True
    validate_inputs: bool = True
    surface_mode: str = 'mixed'
    surface_resolution: int = 96
    inverse_iterations: int = 5
    shadow_samples: int = 8
    geometry_solver: str = 'coarse_to_fine'
    damage_detail: str = 'rich'
    max_font_size: Optional[float] = None
    camera_mode: str = 'full'
    jpeg_enabled: bool = True

    def __post_init__(self):
        for name in ('height', 'width', 'antialias_samples', 'lead_chunk_size',
                     'surface_resolution', 'inverse_iterations', 'shadow_samples'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        for name in ('preserve_extrema', 'validate_inputs', 'jpeg_enabled'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be bool')
        if self.camera_mode not in {'full', 'direct', 'sensor'}:
            raise ValueError('camera_mode must be full, direct or sensor')
        if self.max_font_size is not None and (
                isinstance(self.max_font_size, bool) or not math.isfinite(self.max_font_size)
                or self.max_font_size <= 0):
            raise ValueError('max_font_size must be positive and finite')
        if min(self.height, self.width) < 64:
            raise ValueError('Image dimensions must be at least 64')
        if self.profile not in {'clean', 'scan', 'photo', 'mixed'}:
            raise ValueError('profile must be clean, scan, photo or mixed')
        if not 0 <= self.severity <= 1 or not 0 <= self.jpeg_probability <= 1:
            raise ValueError('Severity and probability must be in [0,1]')
        if not 1 <= self.antialias_samples <= 16 or not 1 <= self.lead_chunk_size <= 15:
            raise ValueError('Use 1..16 antialias samples and 1..15 leads per chunk')
        if not math.isfinite(self.max_duration_s) or self.max_duration_s <= 0:
            raise ValueError('Positive finite maximum duration required')
        if self.surface_mode not in {'mixed','off',*SURFACES}:
            raise ValueError('Invalid surface mode')
        if self.geometry_solver not in {'coarse_to_fine', 'dense'}:
            raise ValueError('geometry_solver must be coarse_to_fine or dense')
        if self.damage_detail not in {'rich', 'edge'}:
            raise ValueError('damage_detail must be rich or edge')
        if not 32 <= self.surface_resolution <= 256 or not 2 <= self.inverse_iterations <= 12 or not 0 <= self.shadow_samples <= 32:
            raise ValueError('Use surface resolution 32..256, inverse iterations 2..12 and shadow samples 0..32')


@dataclass
class RenderBatch:
    image: torch.Tensor
    targets: dict
    metadata: dict
    reference: torch.Tensor
    inverse_field: Optional[torch.Tensor] = None
    canonical_image: Optional[torch.Tensor] = None
    scene: Optional[dict] = None


def _font_atlas(fonts: Optional[Sequence[str]], artifact_texts):
    # The CPU-only initialization is deliberately outside the training hot path.
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    candidates = list(fonts) if fonts else ['DejaVuSans.ttf', 'DejaVuSerif.ttf', 'DejaVuSansMono.ttf']
    found, descriptions = [], []
    for name in candidates:
        try:
            font = ImageFont.truetype(str(name), 24)
        except OSError:
            if fonts:
                raise ValueError(f'Cannot load requested font: {name}')
            continue
        found.append(font)
        descriptions.append(str(name))
    if not found:
        found = [ImageFont.load_default(size=24)]
        descriptions = ['Pillow built-in font']
    tokens = list(LEADS) + [f'{speed:g} mm/s   {gain:g} mm/mV' for speed in SPEEDS[1:] for gain in GAINS[1:]] + ['ECG RECORD', 'RECORD SYNTHETIC'] + ['-'+name for name in LEADS] + list(artifact_texts)
    # Atlas dimensions follow actual token lengths, including user-provided
    # annotation text. A fixed 384px atlas silently truncated long strings.
    atlas_width = max(384, math.ceil(max(font.getlength(token)
                      for font in found for token in tokens)) + 4)
    rows = []
    for font in found:
        for token in tokens:
            image = Image.new('L', (atlas_width, 40), 0)
            ImageDraw.Draw(image).text((1, 1), token, font=font, fill=255, anchor='lt')
            rows.append(torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255)[None])
    return torch.stack(rows), descriptions, tokens


class ECGRenderer(nn.Module):
    """Batched torch renderer. Call .to('cuda') once before using it in a loop.

    stage='trace', photo=False matches v0.5's rectified trace network.
    stage='all' retains independent masks at overlapping lead crossings.
    A seed is a rendering seed, not a patient/partition assignment.
    """
    def __init__(self, config=None, *, fonts=None, paper_textures=None, backgrounds=None,
                 layouts=None, document_assets=None, artifact_texts=('COPY','TEST ECG','CHECK','ARCHIVE','SAMPLE'),
                 compile_raster=False, compile_scene=False):
        super().__init__()
        self.config = config or RenderConfig()
        self._curve_kernel,self._calibration_kernel=raster_kernels(compile_raster)
        self.compile_raster=compile_raster
        self.compile_scene=compile_scene
        if not artifact_texts or any(not isinstance(value,str) or not value or len(value)>80 for value in artifact_texts):
            raise ValueError('artifact_texts must contain nonempty strings with at most 80 characters')
        self.artifact_texts = tuple(artifact_texts)
        self.artifact_offset = 23+len(LEADS)
        atlas, self.fonts, self.tokens = _font_atlas(fonts,self.artifact_texts)
        self.atlas_sha256 = hashlib.sha256(atlas.numpy().tobytes()).hexdigest()
        self.register_buffer('atlas', atlas)
        self.font_count = len(self.fonts)
        self.font_limit = self.config.max_font_size or max(24., .025 * self.config.width)
        # Static construction-time bounds keep the training path on device.
        token_atlas = atlas.reshape(self.font_count, len(self.tokens), *atlas.shape[1:])
        def patch_for(indices):
            occupied = token_atlas[:, indices].amax((0, 1, 2)) > 0
            points = occupied.nonzero()
            if not len(points):
                return (1, 1)
            extent = points.amax(0) + 2
            return tuple(math.ceil(int(value) * self.font_limit / 24) + 1 for value in extent)
        self.lead_patch = patch_for(list(range(12)) + list(range(23, 35)))
        self.header_patch = patch_for(list(range(12, 23)))
        self.artifact_patch = patch_for(list(range(35, len(self.tokens))))
        h, w = self.config.height, self.config.width
        self.register_buffer('xx', torch.arange(w, dtype=torch.float32)[None, None, None, :])
        self.register_buffer('yy', torch.arange(h, dtype=torch.float32)[None, None, :, None])
        self.register_buffer('speed_values', torch.tensor(SPEEDS[1:]))
        self.register_buffer('gain_values', torch.tensor(GAINS[1:]))
        self.register_buffer('palette', torch.tensor([[.84,.38,.40],[.35,.60,.76],[.51,.65,.45],[.59,.59,.59],[.87,.54,.29]]))
        self.register_buffer('age_color', torch.tensor([.02, .07, .20])[None, :, None, None])
        self.register_buffer('pen_color', torch.tensor([.06, .12, .35])[None, :, None, None])
        defaults = standard_layouts()
        if [spec.name for spec in defaults] != LAYOUTS[1:]:
            raise RuntimeError('Layout specification and class table disagree')
        self.layout_specs = list(defaults if layouts is None else layouts)
        if not self.layout_specs or any(not isinstance(spec,LayoutSpec) for spec in self.layout_specs):
            raise ValueError('layouts must contain LayoutSpec objects')
        if len({spec.name for spec in self.layout_specs}) != len(self.layout_specs):
            raise ValueError('Layout names must be unique')
        self.layout_names = [spec.name for spec in self.layout_specs]
        self.instance_count = 15
        default_lookup = {spec.name:spec for spec in defaults}
        tables = {name:[] for name in ['row','col','counts','active','sign','lead','class']}
        rows = []
        for spec in self.layout_specs:
            positions = {token.removeprefix('-'):(r,c,len(row),-1 if token.startswith('-') else 1)
                         for r,row in enumerate(spec.rows) for c,token in enumerate(row)}
            current = [positions.get(name,(0,0,1,1)) for name in LEADS]
            current += [(len(spec.rows)+r,0,1,1) for r in range(3)]
            tables['row'].append([v[0] for v in current])
            tables['col'].append([v[1] for v in current])
            tables['counts'].append([v[2] for v in current])
            tables['sign'].append([v[3] for v in current])
            tables['active'].append([name in positions for name in LEADS]+[r<len(spec.rhythm) for r in range(3)])
            tables['lead'].append(list(range(12))+[LEADS.index(spec.rhythm[r]) if r<len(spec.rhythm) else [1,6,10][r] for r in range(3)])
            tables['class'].append(LAYOUTS.index(spec.name) if default_lookup.get(spec.name)==spec else 0)
            rows.append(len(spec.rows)+len(spec.rhythm))
        for name,values in tables.items():
            self.register_buffer(name+'_table',torch.tensor(values))
        self.register_buffer('rows_table',torch.tensor(rows))
        self.register_buffer('paper_textures', self._check_bank(paper_textures))
        self.register_buffer('backgrounds', self._check_bank(backgrounds))
        self.texture_sha256 = {name: hashlib.sha256(getattr(self, name).detach().cpu().numpy().tobytes()).hexdigest()
                               for name in ['paper_textures', 'backgrounds']}
        self.camera = CameraAppearance(mode=self.config.camera_mode, jpeg_enabled=self.config.jpeg_enabled)
        self.material = PaperMaterial(h, w)
        self.document = DocumentScene(h,w,self.config.surface_resolution,self.config.inverse_iterations,
                                      self.config.shadow_samples,document_assets,self.config.geometry_solver,compile_scene,
                                      damage_detail=self.config.damage_detail)

    @staticmethod
    def _check_bank(bank):
        if bank is None:
            return torch.empty(0, 3, 1, 1)
        value = torch.as_tensor(bank, dtype=torch.float32)
        if value.ndim != 4 or value.shape[0] < 1 or value.shape[1] != 3 or min(value.shape[-2:]) < 8:
            raise ValueError('Texture banks must have shape [K,3,H,W], H/W >=8')
        if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError('Texture values must be finite RGB in [0,1]')
        return value.detach().clone()

    @property
    def device(self):
        return self.xx.device

    def specification(self):
        return {'schema': 'ecg-replot-torch-renderer-v3', 'config': asdict(self.config), 'fonts': self.fonts,
                'signal_units': 'mV', 'signal_order': LEADS, 'font_initialization': 'CPU once',
                'hot_path': 'torch on input device', 'jpeg': 'DCT simulation, not a JPEG bitstream',
                'optical_model': 'linear-light-finite-shutter-gaussian-v2',
                'sensor_noise': 'linear-sensor-independent-read-shot-v2',
                'atlas_sha256': self.atlas_sha256,
                'texture_sha256': self.texture_sha256,
                'paper_texture_count': len(self.paper_textures), 'background_count': len(self.backgrounds),
                'layouts':[asdict(spec) for spec in self.layout_specs], 'instance_count':self.instance_count,
                'artifact_texts':self.artifact_texts,
                'surface_modes':SURFACES,'background_types':BACKGROUNDS,'shadow_types':SHADOWS,
                'document_asset_count':len(self.document.asset_mask), 'document_asset_sha256':self.document.asset_hashes,
                'paper_model':'material-paper-v3-branch-creases',
                'paper_appearance':'linear-absorption-capillary-thermal-v3',
                'paper_damage':'material-fracture-holes-kerfs-corners-v3',
                'scene_illumination':'linear-light-matte-v2',
                'tabletop_shadow':'forward-caster-height-area-light-v2',
                'material_marks':'canonical-shared-inverse-field-v2',
                'supervision_visibility':'conditional-sheet-coverage-v2',
                'procedural_geometry':'developable tangent integration with bounded finite creases, 2.5D',
                'external_geometry':'optional inverse UV/visibility/normal/depth bank'}

    def resolve_parameter(self,key,value):
        """Resolve human-readable constants without reading a device tensor."""
        if key=='layout' and isinstance(value,str):
            if value not in self.layout_names:
                raise ValueError(f'Unknown layout: {value}')
            return self.layout_names.index(value)
        choices={'surface_mode':SURFACES,'background_type':BACKGROUNDS,'object_shadow':SHADOWS}
        if key in choices and isinstance(value,str):
            return choices[key].index(value)
        if key=='rhythm_leads' and isinstance(value,(list,tuple)) and value and isinstance(value[0],str):
            return [LEADS.index(name) for name in value]
        return value

    def sample_parameters(self, batch_size, *, generator=None, photo=True, severity=None):
        b, d = batch_size, self.device
        s = self.config.severity if severity is None else float(severity)
        if not 0 <= s <= 1:
            raise ValueError('Severity must be in [0,1]')
        rand = lambda *shape: torch.rand(shape or (b,), device=d, generator=generator)
        integer = lambda maximum: torch.randint(maximum, (b,), device=d, generator=generator)
        clean = self.config.profile == 'clean'
        if self.config.profile == 'mixed':
            photographic = (rand() > .3).float()
        else:
            photographic = torch.full((b,), float(self.config.profile == 'photo'), device=d)
        photographic *= float(photo and not clean)
        intensity = torch.full((b,), s * float(not clean), device=d)
        geometry = intensity * photographic
        quarter = integer(4) * (geometry > 0).long()
        p = {'layout': integer(len(self.layout_specs)), 'speed_index': integer(3), 'gain_index': integer(3),
             'font': integer(self.font_count), 'palette': integer(len(self.palette)),
             'sequential': rand() > .3, 'quarter': quarter, 'photo': photographic,
             'line_width': .7 + rand() * .9, 'grid_width': .2 + rand() * .45,
             'grid_strength': .24 + rand() * .50, 'minor_strength': .20 + rand() * .35,
             'grid_style':integer(4),
             'grid_present': (rand() > intensity * .12).float(),
             'grid_phase_x': rand(), 'grid_phase_y': rand(),
             'header_present': (rand() > intensity * .22).float(),
             'label_present': (rand(b, 15) > intensity[:, None] * .10).float(),
             'lead_present': (rand(b, 15) > intensity[:, None] * .035).float(),
             'pulse_present':(rand(b,15)>intensity[:,None]*.16).float(),
             'label_below': rand(b, 15) < .18,
             'rhythm_leads':torch.full((b,3),-1,device=d,dtype=torch.long),
             'font_size': (.0075 * self.config.width + rand() * (.006 * self.config.width)).clamp(max=self.font_limit/1.15),
             'paper_tint': rand(b, 3) * .025, 'paper_age': rand() * intensity * .13,
             'paper_noise': rand() * intensity * .012,
             'ink_density': .78 + rand() * .22,
             'ink_color':.025+rand(b,3)*.09,
             'thermal_fade': rand() * intensity * .42,
             'fade_position': rand(), 'fade_width': .03 + rand() * .13,
             'band_strength': rand() * intensity * .018,
             'band_frequency': 20 + rand() * 70,
             'stain_x': rand(), 'stain_y': rand(),
             'stain_strength': rand() * intensity * .12,
             'coffee_strength':rand()*intensity*.20,
             'coffee_radius':.08+rand()*.16,
             'bleed_strength':rand()*intensity*.045,
             'angle': -quarter * (math.pi / 2) + (rand() * 2 - 1) * geometry * .22,
             'zoom': .90 + rand() * .22,
             'translate_x': (rand() * 2 - 1) * geometry * .12,
             'translate_y': (rand() * 2 - 1) * geometry * .10,
             'perspective_x': (rand() * 2 - 1) * geometry * .30,
             'perspective_y': (rand() * 2 - 1) * geometry * .30,
             'radial': (rand() * 2 - 1) * geometry * .06,
             'curl_x': torch.zeros_like(geometry),
             'curl_y': torch.zeros_like(geometry),
             'curl_phase': rand() * (2 * math.pi),
             'crease_position': (rand() * 2 - 1) * .65,
             'crease_cos': (rand() * math.pi).cos(),
             'crease_sin': torch.zeros(b, device=d),
             'crease_warp': torch.zeros_like(geometry),
             'crease_strength': rand() * intensity * .08,
             'exposure': 1 - rand() * intensity * .18,
             'light_x': (rand() * 2 - 1) * geometry * .12,
             'light_y': (rand() * 2 - 1) * geometry * .12,
             'vignette': rand() * geometry * .10,
             'shadow_cos': (rand() * math.pi).cos(),
             'shadow_sin': torch.zeros(b, device=d),
             'shadow_position': (rand() * 2 - 1) * .8,
             'shadow_strength': torch.zeros_like(geometry),
             'white_balance': 1 + (rand(b, 3) * 2 - 1) * intensity[:, None] * .06,
             'blur': .18 + rand() * intensity * .9,
             'motion': rand() * geometry * 2.8,
             'motion_angle': rand() * math.pi,
             'sensor_active': ((intensity > 0).float() if self.config.camera_mode == 'full'
                               else torch.full_like(intensity, float(self.config.camera_mode == 'sensor'))),
             'read_noise': rand() * intensity * .012,
             'shot_noise': rand() * intensity * .008,
             'gamma': 1 + (rand() * 2 - 1) * intensity * .20,
             'jpeg_quality': 98 - rand() * intensity * 57,
             'jpeg_active': (rand() < intensity * self.config.jpeg_probability).float() * float(self.config.jpeg_enabled),
             'occlusion_active': (rand() < intensity * .30).float(),
             'occlusion_x': rand() * .8, 'occlusion_y': .15 + rand() * .7,
             'occlusion_w': .03 + rand() * .16, 'occlusion_h': .015 + rand() * .06,
             'occlusion_color': .65 + rand(b, 3) * .3,
             'pen_active': (rand() < intensity * .22).float(),
             'pen_x': .1 + rand() * .65, 'pen_y': .15 + rand() * .70,
             'pen_phase': rand() * (2 * math.pi),
             'artifact_font':integer(self.font_count),
             'artifact_text':torch.randint(len(self.artifact_texts),(b,4),device=d,generator=generator),
             'artifact_present':(rand(b,4)<intensity[:,None]*.16).float(),
             'artifact_x':rand(b,4)*.72, 'artifact_y':rand(b,4)*.90,
             'artifact_size':((.011+rand(b,4)*.010)*self.config.width).clamp(max=self.font_limit),
             'artifact_angle':(rand(b,4)*2-1)*.40,
             'artifact_color':rand(b,3)*.25}
        p.update(self.document.sample_parameters(b,intensity,geometry,generator=generator,
                                                 surface_mode=self.config.surface_mode))
        p.update(fiber_strength=intensity*.20, foxing_strength=torch.zeros_like(intensity),
                 water_strength=torch.zeros_like(intensity),
                 thermal_darkening_strength=torch.zeros_like(intensity))
        p['crease_sin'] = (1 - p['crease_cos'].square()).clamp_min(0).sqrt()
        p['shadow_sin'] = (1 - p['shadow_cos'].square()).clamp_min(0).sqrt()
        p['zoom'] = torch.where(geometry > 0, p['zoom'], torch.ones_like(p['zoom']))
        return p

    def _instances(self, p, duration):
        b, w, h = len(duration), self.config.width, self.config.height
        layout = p['layout'].long()
        speed, gain = self.speed_values[p['speed_index'].long()], self.gain_values[p['gain_index'].long()]
        # One pixel/mm value governs both axes and the calibration pulse.
        ppm = (w - 1) / (duration * speed + 21.)
        left, right = ppm * 14, w - 1 - ppm * 7
        top = (ppm * 15).clamp(max=h * .17)
        bottom = h - 1 - (ppm * 6).clamp(max=h * .07)
        cols, rows = self.counts_table[layout], self.rows_table[layout]
        cw, ch = (right - left)[:,None] / cols, (bottom - top) / rows
        ri, ci = self.row_table[layout], self.col_table[layout]
        x0 = left[:, None] + ci * cw
        x1 = x0 + cw
        baseline = top[:, None] + (ri + .57) * ch[:, None]
        spans = duration[:,None] / cols
        starts = torch.where(p['sequential'][:, None],ci*spans,0.)
        active = self.active_table[layout].bool() & p['lead_present'].bool()
        lead_ids = self.lead_table[layout].clone()
        lead_ids[:,12:] = torch.where(p['rhythm_leads']>=0,p['rhythm_leads'],lead_ids[:,12:])
        return {'lead_id': lead_ids, 'amplitude_sign':self.sign_table[layout], 'row': ri, 'column': ci,
                'x0': x0, 'x1': x1, 'baseline': baseline, 'start_s': starts,
                'end_s': starts + spans, 'active': active, 'cell_height': ch,
                'ppm': ppm, 'speed': speed, 'gain': gain, 'top': top, 'left': left}

    @staticmethod
    def _sample_signal(signals, times, rates, lengths):
        index = (times * rates[:, None, None]).clamp_min(0)
        index = torch.minimum(index, (lengths - 1)[:, None, None])
        low = index.floor().long()
        high = torch.minimum(low + 1, (lengths - 1)[:, None, None])
        return torch.lerp(signals.gather(2, low), signals.gather(2, high), index - low)

    def _curves(self, signals, rates, lengths, instances, p, stage, return_instances=False,
                return_baselines=True):
        b, _, n = signals.shape
        h, w, samples = self.config.height, self.config.width, self.config.antialias_samples
        trace_target = torch.zeros(b, 12, h, w, device=self.device) if stage in {'trace', 'all'} else None
        baseline_target = torch.zeros_like(trace_target) if trace_target is not None and return_baselines else None
        ink = torch.zeros(b, 1, h, w, device=self.device)
        calibration = torch.zeros_like(ink)
        instance_target = torch.zeros(b,self.instance_count,h,w,device=self.device) if return_instances else None
        dense_masks = trace_target is not None or instance_target is not None
        thickness = p['line_width'][:,None,None,None]/2+.5
        scale = (instances['gain']*instances['ppm'])[:,None]
        slope_time = 1/(instances['speed']*instances['ppm'])[:,None]
        # Keep the original float32 x arithmetic, including non-power-of-two AA.
        axes = torch.stack([self.xx[0,0,0]-.5+sample/samples for sample in range(samples)])
        next_axes = axes+1/samples
        px = (self.xx-axes[:,None,None,:])[:,0]
        source_index = torch.arange(n,device=self.device)[None,None]
        source_times = source_index/rates[:,None,None]
        source_valid = source_index<lengths[:,None,None]
        for first in range(0,self.instance_count,self.config.lead_chunk_size):
            last=min(first+self.config.lead_chunk_size,self.instance_count)
            ids=instances['lead_id'][:,first:last]
            src=signals.gather(1,ids[:,:,None].expand(-1,-1,n))
            src=src*instances['amplitude_sign'][:,first:last,None]
            x0,x1=instances['x0'][:,first:last],instances['x1'][:,first:last]
            base=instances['baseline'][:,first:last]
            start=instances['start_s'][:,first:last]
            active=instances['active'][:,first:last]
            within=(self.xx>=x0[:,:,None,None])&(self.xx<x1[:,:,None,None])&active[:,:,None,None]
            ta=start[:,:,None,None]+(axes[None,None]-x0[:,:,None,None])*slope_time[:,:,None,None]
            tb=start[:,:,None,None]+(next_axes[None,None]-x0[:,:,None,None])*slope_time[:,:,None,None]
            ya=base[:,:,None,None]-self._sample_signal(src,ta.flatten(2),rates,lengths).reshape(b,last-first,samples,w)*scale[:,:,None,None]
            yb=base[:,:,None,None]-self._sample_signal(src,tb.flatten(2),rates,lengths).reshape(b,last-first,samples,w)*scale[:,:,None,None]
            low=high=None
            if self.config.preserve_extrema:
                source_x=x0[:,:,None]+(source_times-start[:,:,None])/slope_time[:,:,None]
                source_y=base[:,:,None]-src*scale[:,:,None]
                included=(source_x>=x0[:,:,None])&(source_x<x1[:,:,None])&source_valid&active[:,:,None]
                column=(source_x+.5).floor().long().clamp(0,w-1)
                low=torch.full((b,last-first,w),float('inf'),device=self.device)
                high=torch.full_like(low,-float('inf'))
                low.scatter_reduce_(2,column,torch.where(included,source_y,float('inf')),reduce='amin',include_self=True)
                high.scatter_reduce_(2,column,torch.where(included,source_y,-float('inf')),reduce='amax',include_self=True)
            mask=self._curve_kernel(ya,yb,px,self.yy,thickness,within,low,high,
                                    self.config.preserve_extrema,not dense_masks)
            ink=torch.maximum(ink,mask.amax(1,keepdim=True) if dense_masks else mask)
            if instance_target is not None:
                instance_target[:,first:last]=mask
            if trace_target is not None:
                index=ids[:,:,None,None].expand_as(mask)
                trace_target.scatter_reduce_(1,index,mask,reduce='amax',include_self=True)
                if baseline_target is not None:
                    baselines=(1-(self.yy-base[:,:,None,None]).abs()).clamp(0,1)*within
                    baseline_target.scatter_reduce_(1,index,baselines,reduce='amax',include_self=True)
            pulse=self._calibration_kernel(base,scale,instances['left'],instances['ppm'],instances['speed'],
                    thickness,p['pulse_present'][:,first:last],active&(instances['column'][:,first:last]==0),self.xx,self.yy)
            calibration=torch.maximum(calibration,pulse)
        targets={} if trace_target is None else {'traces':trace_target}
        if baseline_target is not None:
            targets['baselines']=baseline_target
        if instance_target is not None:
            targets['instances']=instance_target
        return ink,calibration,targets

    def _grid(self, instances, p, need_vertices):
        ppm = instances['ppm'][:, None, None, None]
        x = self.xx - p['grid_phase_x'][:, None, None, None] * ppm
        y = self.yy - p['grid_phase_y'][:, None, None, None] * ppm
        distance = lambda value, step: ((value + step / 2).remainder(step) - step / 2).abs()
        dx, dy = distance(x, ppm), distance(y, ppm)
        mx, my = distance(x, 5 * ppm), distance(y, 5 * ppm)
        width = p['grid_width'][:, None, None, None]
        minor = torch.maximum((width / 2 + .5 - dx).clamp(0, 1), (width / 2 + .5 - dy).clamp(0, 1))
        major = torch.maximum((width + .5 - mx).clamp(0, 1), (width + .5 - my).clamp(0, 1))
        style = p['grid_style'][:,None,None,None]
        minor_dots = (width+.65-(dx.square()+dy.square()).sqrt()).clamp(0,1)
        major_dots = (width+1.0-(mx.square()+my.square()).sqrt()).clamp(0,1)
        minor = torch.where(style>=2,minor_dots,minor)*(style!=1)
        major = torch.where(style==3,major_dots,major)
        region = self.yy >= instances['top'][:, None, None, None] * .8
        visible = region * p['grid_present'][:, None, None, None]
        grid = torch.maximum(minor, major) * visible
        alpha = (major + minor * p['minor_strength'][:, None, None, None]).clamp(0, 1) * p['grid_strength'][:, None, None, None] * visible
        vertices = None
        if need_vertices:
            sigma = (ppm * .23).clamp(.55, 1.1)
            main = torch.exp(-(mx.square() + my.square()) / (2 * sigma.square()))
            secondary = torch.exp(-(dx.square() + dy.square()) / (2 * sigma.square())) * (1 - main) * (style!=1)
            vertices = torch.cat([main, secondary], 1) * visible
        return grid, alpha, vertices

    def _stamp(self, tokens, x, y, size, present, channels=None, patch=None, rotation=None):
        """Rasterize small glyph patches then max-splat, avoiding full-page atlases."""
        b, jobs = tokens.shape
        ph, pw = self.lead_patch if patch is None else patch
        xx = torch.arange(pw, device=self.device)[None, None, :].float()
        yy = torch.arange(ph, device=self.device)[None, :, None].float()
        scale = size.reshape(b * jobs, 1, 1) / 24
        gx = 2 * ((xx + .5) / scale) / self.atlas.shape[-1] - 1
        gy = 2 * ((yy + .5) / scale) / self.atlas.shape[-2] - 1
        field = torch.stack([gx.expand(-1, ph, -1), gy.expand(-1, -1, pw)], -1)
        source = self.atlas[tokens.reshape(-1)]
        glyph = F.grid_sample(source, field, align_corners=False)[:, 0].reshape(b, jobs, ph, pw)
        glyph *= present[..., None, None]
        sx = torch.arange(pw,device=self.device)[None,None,None,:]
        sy = torch.arange(ph,device=self.device)[None,None,:,None]
        if rotation is not None:
            c,s = rotation.cos()[...,None,None],rotation.sin()[...,None,None]
            sx,sy = sx*c-sy*s,sx*s+sy*c
        ix = (x[...,None,None]+sx).round().long()
        iy = (y[...,None,None]+sy).round().long()
        inside = (ix >= 0) & (ix < self.config.width) & (iy >= 0) & (iy < self.config.height)
        count = 1 if channels is None else 12
        offsets = torch.arange(b, device=self.device)[:, None, None, None] * count
        if channels is not None:
            offsets = offsets + channels[..., None, None]
        index = (offsets * self.config.height + iy.clamp(0, self.config.height - 1)) * self.config.width + ix.clamp(0, self.config.width - 1)
        output = torch.zeros(b * count * self.config.height * self.config.width, device=self.device)
        output.scatter_reduce_(0, index.expand_as(glyph).reshape(-1), (glyph * inside).reshape(-1), reduce='amax', include_self=True)
        return output.reshape(b, count, self.config.height, self.config.width)

    def _text(self, instances, p, need_labels, need_units=True):
        b = len(p['font'])
        lead_tokens = torch.where(instances['amplitude_sign']<0,23+instances['lead_id'],instances['lead_id'])
        tokens = p['font'][:, None] * len(self.tokens) + lead_tokens
        x = instances['x0'] + 2
        y = torch.where(p['label_below'], instances['baseline'] + instances['cell_height'][:, None] * .12,
                        instances['baseline'] - instances['cell_height'][:, None] * .39)
        size = p['font_size'][:, None].expand(b, self.instance_count)
        present = instances['active'] * p['label_present']
        label_masks = self._stamp(tokens, x, y, size, present, instances['lead_id'] if need_labels else None)
        label_ink = label_masks.amax(1, keepdim=True) if need_labels else label_masks
        units = 12 + p['speed_index'].long() * 3 + p['gain_index'].long()
        tokens = p['font'][:, None] * len(self.tokens) + torch.stack([units, torch.full_like(units, 21)], 1)
        header_x = torch.stack([instances['left'], torch.full_like(instances['left'], self.config.width * .68)], 1)
        header_y = torch.full((b, 2), self.config.height * .022, device=self.device)
        header_size = (p['font_size'] * 1.15)[:, None].expand(b, 2)
        headers = self._stamp(tokens, header_x, header_y, header_size,
                              torch.stack([p['header_present'], torch.ones_like(p['header_present'])], 1), patch=self.header_patch)
        # Keep the unit glyph support separately to test cropping/occlusion.
        unit_mask = self._stamp(tokens[:, :1], header_x[:, :1], header_y[:, :1], header_size[:, :1], p['header_present'][:, None], patch=self.header_patch) if need_units else None
        return torch.maximum(label_ink, headers), label_masks if need_labels else None, unit_mask

    def _texture(self, bank, b, generator, paper):
        h, w = self.config.height, self.config.width
        if len(bank):
            ids = torch.randint(len(bank), (b,), device=self.device, generator=generator)
            crop_scale = .55 + torch.rand(b, device=self.device, generator=generator) * .40
            offset = (torch.rand(b, 2, device=self.device, generator=generator) * 2 - 1) * (1 - crop_scale[:, None])
            gx = (2 * (self.xx[:, 0] + .5) / w - 1).expand(b, h, w) * crop_scale[:, None, None] + offset[:, 0, None, None]
            gy = (2 * (self.yy[:, 0] + .5) / h - 1).expand(b, h, w) * crop_scale[:, None, None] + offset[:, 1, None, None]
            return F.grid_sample(bank[ids], torch.stack([gx, gy], -1), align_corners=False), ids
        small = torch.rand(b, 1, 24, 24, device=self.device, generator=generator)
        texture = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False).expand(-1, 3, -1, -1)
        if paper:
            return .992+(texture-.5)*.003,torch.full((b,),-1,device=self.device,dtype=torch.long)
        raise RuntimeError('Procedural backgrounds are generated by DocumentScene')

    def _field(self, p):
        b, h, w = len(p['angle']), self.config.height, self.config.width
        x = (self.xx[:, 0] - (w - 1) / 2) / (w / 2)
        y = (self.yy[:, 0] - (h - 1) / 2) / (w / 2)
        x = x.expand(b, h, w) - p['translate_x'][:, None, None]
        y = y.expand(b, h, w) - p['translate_y'][:, None, None]
        radius = x.square() + y.square()
        factor = 1 + p['radial'][:, None, None] * radius
        x, y = x * factor, y * factor
        denominator = 1 + p['perspective_x'][:, None, None] * x + p['perspective_y'][:, None, None] * y
        if self.config.validate_inputs:
            torch._assert_async((denominator > .08).all(),
                                'Perspective crosses or approaches the camera horizon')
        x, y = x / denominator, y / denominator
        c, s = p['angle'].cos()[:, None, None], p['angle'].sin()[:, None, None]
        fit = torch.where(p['quarter'].remainder(2) == 1, torch.full_like(p['zoom'], min(h / w, w / h)), torch.ones_like(p['zoom']))
        zoom = (p['zoom'] * fit)[:, None, None]
        u, v = (c * x + s * y) / zoom, (-s * x + c * y) / zoom
        phase = p['curl_phase'][:, None, None]
        u, v = u + p['curl_x'][:, None, None] * torch.sin(v * math.pi * 2 + phase), v + p['curl_y'][:, None, None] * torch.sin(u * math.pi * 2 + phase)
        crease = u * p['crease_cos'][:, None, None] + v * p['crease_sin'][:, None, None] - p['crease_position'][:, None, None]
        displacement = p['crease_warp'][:, None, None] * torch.tanh(crease / .035)
        u = u + displacement * p['crease_cos'][:, None, None]
        v = v + displacement * p['crease_sin'][:, None, None]
        return torch.stack([u, v * w / h], -1)

    @torch.no_grad()
    def forward(self, signals, fs, *, lengths=None, stage='all', photo=True, generator=None,
                parameters: Optional[Mapping] = None, severity=None, return_field=False, return_canonical=False,
                return_scene=False, return_instances=False, return_baselines=True,
                return_instance_boxes=True, as_texture=False, return_visibility_fraction=None):
        # Keep geometric inversion, glyphs and radiometry in float32 even when
        # the caller trains its image network inside an AMP context.
        if self.atlas.dtype!=torch.float32:
            raise ValueError('Keep renderer buffers float32. Use autocast for the downstream image network')
        with torch.autocast(device_type=signals.device.type,enabled=False):
            return self._render(signals,fs,lengths=lengths,stage=stage,photo=photo,generator=generator,
                                parameters=parameters,severity=severity,return_field=return_field,
                                return_canonical=return_canonical,return_scene=return_scene,return_instances=return_instances,
                                return_baselines=return_baselines,return_instance_boxes=return_instance_boxes,
                                as_texture=as_texture,return_visibility_fraction=return_visibility_fraction)

    def _render(self, signals, fs, *, lengths=None, stage='all', photo=True, generator=None,
                parameters: Optional[Mapping] = None, severity=None, return_field=False, return_canonical=False,
                return_scene=False, return_instances=False, return_baselines=True,
                return_instance_boxes=True, as_texture=False, return_visibility_fraction=None):
        if type(photo) is not bool or type(as_texture) is not bool:
            raise ValueError('photo and as_texture must be bool')
        if as_texture and photo:
            raise ValueError('as_texture=True requires photo=False')
        if return_visibility_fraction is None:
            return_visibility_fraction = return_instances
        if type(return_visibility_fraction) is not bool or (return_visibility_fraction and not return_instances):
            raise ValueError('Visibility fractions require return_instances=True')
        if stage not in {'image','geometry', 'trace', 'metadata', 'all'}:
            raise ValueError('stage must be image, geometry, trace, metadata or all')
        if signals.ndim != 3 or signals.shape[0] < 1 or signals.shape[1] != 12 or signals.shape[2] < 2:
            raise ValueError('Signals must have shape [B,12,T], T>=2, in standard lead order')
        if signals.device != self.device or not signals.is_floating_point():
            raise ValueError('Floating signals and renderer must be on the same device')
        b, _, n = signals.shape
        rates = torch.as_tensor(fs, device=self.device, dtype=torch.float32)
        if rates.ndim > 1 or rates.numel() not in (1, b):
            raise ValueError('fs must be a scalar or a vector with one sample rate per record')
        rates = rates.reshape(-1).expand(b)
        raw_lengths = lengths
        lengths = torch.full((b,), n, device=self.device, dtype=torch.long) if lengths is None else torch.as_tensor(lengths, device=self.device)
        if not isinstance(raw_lengths, torch.Tensor) and lengths.is_floating_point():
            # Validate Python float fractions before default float32 inference
            # can round them to an apparently integral source length.
            lengths = torch.as_tensor(raw_lengths, device=self.device, dtype=torch.float64)
        if lengths.shape != (b,):
            raise ValueError('lengths must have shape [B]')
        if lengths.is_complex():
            raise ValueError('lengths must contain real integer sample counts')
        working_signals = signals.float()
        if self.config.validate_inputs:
            # CUDA assertion stays on device, without a per-batch .item() sync.
            torch._assert_async(torch.isfinite(signals).all(), 'Nonfinite ECG signals')
            torch._assert_async(torch.isfinite(working_signals).all(),
                                'ECG signals must be finite in float32 working precision')
            torch._assert_async((torch.isfinite(rates) & (rates > 0)).all(), 'Positive finite sample rates required')
            torch._assert_async(((lengths >= 2) & (lengths <= n) & (lengths == lengths.long())).all(),
                                'Invalid source lengths: finite integer sample counts required')
        lengths = lengths.long()
        p = self.sample_parameters(b, generator=generator, photo=photo, severity=severity)
        sampled_jitter = p['angle'] + p['quarter'] * (math.pi / 2)
        if parameters:
            unknown = set(parameters) - p.keys()
            if unknown:
                raise ValueError(f'Unknown rendering parameters: {sorted(unknown)}')
            for key, value in parameters.items():
                raw_value = self.resolve_parameter(key,value)
                value = torch.as_tensor(raw_value, device=self.device)
                if not p[key].is_floating_point() and not isinstance(raw_value, torch.Tensor) and value.is_floating_point():
                    value = torch.as_tensor(raw_value, device=self.device, dtype=torch.float64)
                if value.is_complex():
                    raise ValueError(f'Real rendering parameter required: {key}')
                if p[key].dtype == torch.bool:
                    torch._assert_async(((value == 0) | (value == 1)).all(),
                                        f'Boolean rendering parameter must be 0 or 1: {key}')
                if not p[key].is_floating_point() and p[key].dtype != torch.bool and value.is_floating_point():
                    torch._assert_async((torch.isfinite(value) & (value == value.round())).all(),
                                        f'Integer rendering parameter required: {key}')
                p[key] = value.to(dtype=p[key].dtype).expand_as(p[key])
            classes_valid = ((p['layout'] >= 0) & (p['layout'] < len(self.layout_specs))
                             & (p['speed_index'] >= 0) & (p['speed_index'] < 3)
                             & (p['gain_index'] >= 0) & (p['gain_index'] < 3)
                             & (p['quarter'] >= 0) & (p['quarter'] < 4) & (p['zoom'] > 0))
            torch._assert_async(classes_valid.all(), 'Invalid layout, units, orientation or zoom')
            torch._assert_async(torch.isfinite(torch.cat([value.reshape(-1).float() for value in p.values()])).all(), 'Nonfinite rendering parameters')
            alpha_keys = ['lead_present', 'label_present', 'pulse_present', 'header_present', 'grid_present',
                          'grid_strength', 'minor_strength', 'ink_density', 'thermal_fade', 'sensor_active',
                          'jpeg_active', 'occlusion_active', 'pen_active', 'artifact_present',
                          'paper_age', 'stain_strength', 'coffee_strength', 'fiber_strength',
                          'foxing_strength', 'water_strength', 'thermal_darkening_strength',
                          'edge_wear_strength', 'hole_active', 'cut_active', 'missing_corner_active',
                          'tear_raggedness']
            alpha = torch.cat([p[key].reshape(-1) for key in alpha_keys])
            torch._assert_async(((alpha >= 0) & (alpha <= 1)).all(),
                                'Rendering presence, opacity and ink density must be in [0,1]')
            for key in ['lead_present','photo','tear_active','hole_active','cut_active','missing_corner_active']:
                torch._assert_async(((p[key]==0)|(p[key]==1)).all(),f'{key} must be binary')
            torch._assert_async(((p['font_size'] > 0) & (p['font_size'] * 1.15 <= self.font_limit)).all(),
                                'font_size exceeds the configured max_font_size including header scale')
            torch._assert_async(((p['artifact_size'] > 0) & (p['artifact_size'] <= self.font_limit)).all(),
                                'artifact_size exceeds the configured max_font_size')
            torch._assert_async(((p['coffee_radius']>=.01)&(p['coffee_radius']<=.5)).all(),
                                'Coffee radius must be in [.01,.5]')
            if self.config.damage_detail == 'edge':
                advanced = torch.cat([p[key].reshape(-1) for key in
                    ['hole_active','cut_active','missing_corner_active','edge_wear_strength']])
                torch._assert_async((advanced==0).all() & (p['tear_count']<=1).all(),
                                    'Advanced damage requires damage_detail=rich')
            torch._assert_async(((p['rhythm_leads']>=-1)&(p['rhythm_leads']<12)).all(),'Invalid rhythm lead IDs')
            for key in ['fade_width','fold_width','fold_span','fold_length','corner_size','object_shadow_size','object_shadow_softness','background_scale']:
                torch._assert_async((p[key]>0).all(),f'Positive {key} required')
            for key in ['paper_color','ink_color','artifact_color']:
                torch._assert_async(((p[key]>=0)&(p[key]<=1)).all(),f'RGB {key} must be in [0,1]')
            torch._assert_async(((p['surface_strength']>=0)&(p['surface_strength']<=1)).all(),'Surface strength must be in [0,1]')
            torch._assert_async(((p['crease_count']>=0)&(p['crease_count']<=8)).all(),'Crease count must be in 0..8')
            for key, maximum in [('tear_count',4), ('missing_corner_index',3)]:
                torch._assert_async(((p[key]>=0)&(p[key]<=maximum)).all(),f'{key} outside supported range')
            for key in ['hole_center','cut_center']:
                torch._assert_async((p[key].abs()<=1).all(),f'{key} must use normalized sheet coordinates')
            for key, maximum, active in [
                    ('hole_radius',.25,p['hole_active'][...,None]),
                    ('cut_length',1.,p['cut_active']), ('cut_width',.05,p['cut_active']),
                    ('missing_corner_size',.6,p['missing_corner_active'][:,None])]:
                size_valid = (p[key]>=0)&(p[key]<=maximum)&((p[key]>0)|(active==0))
                torch._assert_async(size_valid.all(),f'Invalid {key}')
            torch._assert_async(((p['edge_wear_depth']>=0)&(p['edge_wear_depth']<=.05)).all(),
                                'Edge wear depth must be in [0,.05]')
            torch._assert_async((p['paper_gap']>=0).all(),'Paper gap must be nonnegative')
            for key in ['read_noise','shot_noise','exposure','white_balance']:
                torch._assert_async((p[key]>=0).all(),f'Nonnegative {key} required')
            torch._assert_async((p['gamma']>0).all(),'Positive gamma required')
            torch._assert_async(((p['blur']>=0)&(p['blur']<=1.1)).all(),
                                'Optical blur sigma must be in [0,1.1] pixels for the finite PSF')
            torch._assert_async(((p['motion']>=0)&(p['motion']<=3.2)).all(),
                                'Optical motion must be in [0,3.2] pixels for the finite PSF')
            torch._assert_async(((p['artifact_text']>=0)&(p['artifact_text']<len(self.artifact_texts))).all(),'Invalid artifact text')
            for key,count in [('font',self.font_count),('artifact_font',self.font_count),('palette',len(self.palette)),('grid_style',4),
                              ('surface_mode',len(SURFACES)),('background_type',len(BACKGROUNDS)),
                              ('object_shadow',len(SHADOWS)),('paper_palette',len(self.document.paper_colors)),
                              ('asset_index',max(1,len(self.document.asset_mask)))]:
                torch._assert_async(((p[key]>=0)&(p[key]<count)).all(),f'Invalid {key}')
        if parameters and 'quarter' in parameters and 'angle' not in parameters:
            p['angle'] = -p['quarter'] * (math.pi / 2) + sampled_jitter
        if parameters and 'angle' in parameters:
            derived_quarter = (-p['angle'] / (math.pi / 2)).round().long().remainder(4)
            if 'quarter' in parameters and photo:
                torch._assert_async((derived_quarter == p['quarter']).all(),
                                    'quarter contradicts the absolute angle')
            p['quarter'] = derived_quarter
        # The public photo parameter is a per-record geometric capture gate.
        # Optical sensor controls remain independent, allowing flatbed scans.
        if parameters and 'photo' in parameters:
            enabled = p['photo'].bool()
            for key in ['angle','quarter','translate_x','translate_y','perspective_x','perspective_y',
                        'radial','curl_x','curl_y','crease_warp','surface_strength','asset_active']:
                p[key] = torch.where(enabled, p[key], torch.zeros_like(p[key]))
            p['zoom'] = torch.where(enabled, p['zoom'], torch.ones_like(p['zoom']))
            p['relief_view'] = p['relief_view'] * enabled[:, None]
        if not photo:
            # Canonical trace supervision never contains residual camera warps.
            for key in ['angle', 'quarter', 'translate_x', 'translate_y', 'perspective_x', 'perspective_y', 'radial', 'curl_x', 'curl_y', 'crease_warp']:
                p[key] = torch.zeros_like(p[key])
            p['zoom'] = torch.ones_like(p['zoom'])
            p['relief_view'] = torch.zeros_like(p['relief_view'])
            p['surface_strength'] = torch.zeros_like(p['surface_strength'])
            p['asset_active'] = torch.zeros_like(p['asset_active'])
            p['photo'] = torch.zeros_like(p['photo'])
        duration = ((lengths - 1) / rates).clamp(max=self.config.max_duration_s)
        instances = self._instances(p, duration)
        if self.config.validate_inputs:
            # Raster distances and segment projections square pixel offsets.
            # Bound arithmetic by working precision, without imposing a clinical
            # amplitude cap or silently clipping the supplied physiological data.
            pixel_extent = (working_signals.abs().amax((1, 2)) * instances['gain'] * instances['ppm']
                            + max(self.config.height, self.config.width))
            torch._assert_async((pixel_extent < math.sqrt(torch.finfo(torch.float32).max / 16)).all(),
                                'ECG pixel coordinates exceed safe float32 raster precision')
        ink, pulse, targets = self._curves(working_signals, rates, lengths, instances, p, stage,return_instances,return_baselines)
        source_instance_area = targets['instances'].sum((-1, -2)) if return_visibility_fraction else None
        grid, grid_alpha, vertices = self._grid(instances, p, stage in {'geometry', 'all'})
        text, labels, unit_mask = self._text(instances, p, stage in {'metadata', 'all'},stage!='image')
        paper, paper_id = self._texture(self.paper_textures, b, generator, True)
        paper *= self.document.paper_colors[p['paper_palette'].long()][:,:,None,None]*p['paper_color'][:,:,None,None]
        paper = paper - p['paper_tint'][:, :, None, None]
        paper += torch.randn(paper.shape, device=self.device, generator=generator) * p['paper_noise'][:, None, None, None]
        material = self.material(p, generator)
        factor = material['substrate_factor']
        paper = torch.where(factor != 1, linear_to_srgb(srgb_to_linear(paper.clamp(0,1))*factor), paper)
        reverse_ink = F.avg_pool2d(ink.flip(-1),5,1,2)
        paper -= reverse_ink*p['bleed_strength'][:,None,None,None]
        color = self.palette[p['palette'].long()][:, :, None, None]
        canonical = torch.lerp(paper.clamp(0, 1), color, grid_alpha)
        fade = material['print_retention']
        print_alpha = torch.maximum(torch.maximum(ink, pulse), text) * fade * p['ink_density'][:, None, None, None]
        canonical = torch.lerp(canonical, p['ink_color'][:,:,None,None].expand_as(canonical), print_alpha)
        canonical += p['band_strength'][:, None, None, None] * torch.sin(self.yy / self.config.height * p['band_frequency'][:, None, None, None] * (2 * math.pi))
        # Pen and printed stamps are attached to paper material coordinates.
        # They undergo exactly the same inverse warp as the ECG and cannot
        # float at fixed camera pixels while the page bends or rotates.
        xx, yy = self.xx / self.config.width, self.yy / self.config.height
        def column(name):
            return p[name][:, None, None, None]
        local_x = (xx - column('pen_x')) / .17
        pen_y = column('pen_y') + .023 * torch.sin(local_x * 16 + column('pen_phase')) + .015 * local_x
        pen = (1 - ((yy - pen_y) * self.config.height).abs() / 1.4).clamp(0, 1)
        pen *= ((local_x >= 0) & (local_x <= 1)) * column('pen_active')
        canonical = torch.lerp(canonical, self.pen_color, pen)
        tokens = p['artifact_font'][:,None]*len(self.tokens)+self.artifact_offset+p['artifact_text'].long()
        artifacts = self._stamp(tokens,p['artifact_x']*self.config.width,p['artifact_y']*self.config.height,
                                p['artifact_size'],p['artifact_present'],patch=self.artifact_patch,rotation=p['artifact_angle'])
        canonical = torch.lerp(canonical,p['artifact_color'][:,:,None,None].expand_as(canonical),artifacts)
        # This chronology models liquid deposited after ECG printing and marks.
        # Transmission changes observation contrast, never geometric ownership.
        transmission = material['deposit_transmission']
        canonical = torch.where(transmission != 1,
            linear_to_srgb(srgb_to_linear(canonical.clamp(0,1))*transmission), canonical)
        material_valid = ((pen < .15) & (artifacts < .15)).float()
        # Crease illumination comes from the material mesh. No independent
        # painted highlight is placed on otherwise flat paper.
        if vertices is not None:
            targets.update(vertices=vertices, grid=grid)
        if labels is not None:
            targets['labels'] = labels
        if not photo and stage != 'image':
            # Rectified substrate support lets the offline 3D backend remove
            # the same physical paper, independently of its printed texture.
            targets['paper_support'] = self.document.paper_mask(self._field(p), p)
        field = self._field(p) if photo or return_field else None
        scene = None
        if photo:
            if self.config.surface_mode != 'off' or len(self.document.asset_mask):
                scene = self.document.deform(field,p,generator,diagnostics=return_scene)
                field = scene['field']
            # Fractional coverage antialiases the sheet silhouette. The same
            # coverage gates targets, cast shadows and RGB compositing.
            valid = F.grid_sample(torch.ones_like(ink), field, align_corners=False)
            frame_coverage = valid.clone() if stage != 'image' or return_instances else None
            image = F.grid_sample(srgb_to_linear(canonical), field, align_corners=False, padding_mode='border')
            if scene is None:
                scene = {'field':field,'solver_valid':torch.ones_like(valid),
                         'normals':torch.cat([torch.zeros_like(valid),torch.zeros_like(valid),torch.ones_like(valid)],1),
                         'height':torch.zeros_like(valid),'illumination':torch.ones_like(image),
                         'asset_id':torch.full((b,),-1,device=self.device,dtype=torch.long)}
            valid *= scene['solver_valid']*self.document.paper_mask(field,p)
            # Material colours and imported RGB textures are encoded sRGB.
            # Irradiance multiplies radiance, never display code values. Keep
            # highlights until the camera sensor performs its own clipping.
            image = image*scene['illumination']
            if len(self.backgrounds):
                background,background_id = self._texture(self.backgrounds,b,generator,False)
            else:
                background = self.document.background(p,generator)
                background_id = torch.full((b,),-1,device=self.device,dtype=torch.long)
            scene['cast_shadow'] = self.document.cast_shadow(valid,scene,p)
            background = srgb_to_linear(background)*(1-scene['cast_shadow'])
            image = torch.lerp(background,image,valid.clamp(0,1))
            scene['paper_mask'] = valid.clone()
            scene['object_shadow'] = self.document.object_shadow(p)
            image = linear_to_srgb(image*(1-scene['object_shadow']))
            targets = {name: F.grid_sample(value, field, align_corners=False) for name, value in targets.items()}
            if frame_coverage is not None:
                targets['frame_coverage'] = frame_coverage
            unit_support = F.grid_sample(unit_mask, field, align_corners=False) if unit_mask is not None else None
            material_valid = F.grid_sample(material_valid, field, align_corners=False, padding_mode='border')
        else:
            valid, image, unit_support = torch.ones_like(ink), canonical, unit_mask
            if not as_texture:
                support = self.document.paper_mask(self._field(p), p)
                valid = support.clone()
                image = torch.lerp(torch.ones_like(canonical), canonical, support)
            background_id = torch.full((b,), -1, device=self.device, dtype=torch.long)
        # This legacy opaque rectangle represents a scene/camera occluder.
        # Unlike pen and stamp ink, it intentionally stays in output pixels.
        cover = ((xx >= column('occlusion_x')) & (xx <= column('occlusion_x') + column('occlusion_w'))
                 & (yy >= column('occlusion_y')) & (yy <= column('occlusion_y') + column('occlusion_h'))).float() * column('occlusion_active')
        image = torch.lerp(image, p['occlusion_color'][:, :, None, None], cover)
        valid *= (1-cover)*material_valid
        if stage!='image':
            targets['valid'] = valid
        area = material_pixel_area(field, self.config.height, self.config.width) if photo and (
            unit_mask is not None or return_visibility_fraction) else 1.
        visibility = _conditional_visibility(valid, targets.get('frame_coverage'))
        if return_visibility_fraction:
            visible_area = (targets['instances'] * visibility * area).sum((-1, -2))
            targets['instance_visible_fraction'] = torch.where(source_instance_area > 0,
                visible_area / source_instance_area.clamp_min(torch.finfo(torch.float32).tiny), 0).clamp(0, 1)
        unit_evidence = None
        if unit_mask is not None:
            original_units = unit_mask.sum((-1, -2, -3))
            # Compare observed and uncropped support under the same resampling field.
            observed_units = (unit_support * visibility * area).sum((-1, -2, -3))
            unit_evidence = (observed_units / original_units.clamp_min(1)) > .75
            unit_evidence &= p['header_present'].bool()
        if stage in {'metadata', 'all'}:
            targets.update(orientation=p['quarter'].long(), layout=self.class_table[p['layout'].long()],
                           speed=torch.where(unit_evidence, p['speed_index'].long() + 1, 0),
                           gain=torch.where(unit_evidence, p['gain_index'].long() + 1, 0))
        if self.config.profile == 'clean' and not parameters:
            image = (image.clamp(0, 1) * 255).round() / 255
        else:
            image = self.camera(image, p, generator)
        metadata = {**instances, 'rectified_mode': 'texture' if as_texture else 'document',
                    'sample_rate_hz': rates, 'source_length': lengths, 'duration_s': duration,
                    'canonical_pixels_per_second': instances['ppm'] * instances['speed'],
                    'canonical_millivolts_per_pixel': (instances['ppm'] * instances['gain']).reciprocal(),
                    'source_samples_per_canonical_pixel': rates / (instances['ppm'] * instances['speed']),
                    'layout_class': self.class_table[p['layout'].long()], 'orientation_class': p['quarter'].long(),
                    'sequential': p['sequential'], 'unit_text_visible': unit_evidence,
                    'paper_texture_id': paper_id, 'background_id': background_id,
                    'document_asset_id':scene['asset_id'] if scene else torch.full((b,),-1,device=self.device,dtype=torch.long),
                    'parameters': p}
        if return_instances and return_instance_boxes:
            visible = targets['instances'] * _conditional_visibility(valid, targets.get('frame_coverage'))
            occupied = visible>.05
            columns = occupied.any(-2)
            rows = occupied.any(-1)
            xs = torch.arange(self.config.width,device=self.device)[None,None]
            ys = torch.arange(self.config.height,device=self.device)[None,None]
            x0 = torch.where(columns,xs,self.config.width).amin(-1)
            x1 = torch.where(columns,xs+1,0).amax(-1)
            y0 = torch.where(rows,ys,self.config.height).amin(-1)
            y1 = torch.where(rows,ys+1,0).amax(-1)
            has_pixels = (x1>x0)&(y1>y0)
            boxes = torch.stack([x0,y0,x1,y1],-1)
            metadata['instance_boxes_xyxy'] = torch.where(has_pixels[...,None],boxes,-1)
            metadata['instance_visible'] = has_pixels
        return RenderBatch(image, targets, metadata, signals, field if return_field else None,
                           canonical if return_canonical else None,scene if return_scene else None)


