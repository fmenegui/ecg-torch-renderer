"""Torch height-field documents, coherent lighting and tabletop scenes.

Procedural surfaces use a 2.5D relief model, not a cloth or path-tracing solver.
An optional bank of externally rendered inverse UV/visibility/normal/depth maps
supports fully 3D shapes and self-occlusion without rendering a mesh in the loop.
All random sampling, inverse solving, shading and composition stay on device.
"""
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from .gpu_paper import PaperSurface, derivatives, newton_step


SURFACES = ('flat','curl','fold','accordion','crumpled','wrinkled','rolled','corner')
BACKGROUNDS = ('wood','marble','laminate','fabric','metal','clinical','plain','leather')
SHADOWS = ('none','edge','phone','hand','blinds','window','ellipse')


@dataclass
class DocumentAssets:
    """All maps describe the same externally rendered camera and paper.

    inverse_field: [K,H,W,2], output pixel -> original ECG paper, normalized
    for torch grid_sample(align_corners=False). mask: [K,1,H,W], visible paper
    only. normals: [K,3,H,W], unit vectors in the bank camera frame. height:
    [K,1,H,W], nonnegative relief in half-page-width units. Optional shading
    multiplies sampled illumination. It can preserve offline path-traced
    shadows. No document text or patient information should be in these maps.
    """
    inverse_field: torch.Tensor
    mask: torch.Tensor
    normals: torch.Tensor = None
    height: torch.Tensor = None
    shading: torch.Tensor = None

    @classmethod
    def from_npz(cls, path):
        # Asset decoding is construction-time CPU work, never a batch operation.
        import numpy as np
        with np.load(path, allow_pickle=False) as source:
            if str(source['schema']) != 'ecg-document-assets-v1':
                raise ValueError('Explicit ecg-document-assets-v1 schema required')
            return cls(**{name: torch.from_numpy(source[name].copy())
                          for name in ['inverse_field','mask','normals','height','shading'] if name in source})


class DocumentScene(nn.Module):
    def __init__(self, height, width, resolution=96, inverse_iterations=5,
                 shadow_samples=8, assets=None, geometry_solver='coarse_to_fine', compile_scene=False,
                 damage_detail='rich'):
        super().__init__()
        self.height, self.width, self.aspect = height, width, height / width
        self.resolution = resolution
        self.inverse_iterations, self.shadow_samples = inverse_iterations, shadow_samples
        if geometry_solver not in {'coarse_to_fine', 'dense'}:
            raise ValueError('geometry_solver must be coarse_to_fine or dense')
        if not isinstance(compile_scene, bool):
            raise ValueError('compile_scene must be a boolean')
        if damage_detail not in {'rich', 'edge'}:
            raise ValueError('damage_detail must be rich or edge')
        self.damage_detail = damage_detail
        self.geometry_solver = geometry_solver
        self.compile_scene = compile_scene
        self._newton_step = torch.compile(newton_step, fullgraph=True, dynamic=True) if compile_scene else newton_step
        rows = max(24, round(resolution * self.aspect))
        self.surface_model = PaperSurface(rows, resolution, self.aspect, compile_scene)
        self._illumination_kernel = torch.compile(self.illumination, fullgraph=True, dynamic=True) if compile_scene else self.illumination
        x, y = self.surface_model.material[0]
        self.register_buffer('x',x[None,None].clone())
        self.register_buffer('y',y[None,None].clone())
        by, bx = torch.meshgrid((torch.arange(rows * 2) + .5) * (2 * self.aspect / (rows * 2)) - self.aspect,
                                (torch.arange(resolution * 2) + .5) * (2 / (resolution * 2)) - 1, indexing='ij')
        self.register_buffer('background_xy',torch.stack((bx, by), 0)[None])
        self.register_buffer('paper_colors',torch.tensor([
            [.985,.980,.970],[.990,.978,.965],[.975,.962,.935],[.970,.945,.900],
            [.960,.960,.950],[.985,.980,.950],[.935,.930,.910],[.910,.875,.805]]))
        self.register_buffer('table_colors',torch.tensor([
            [.42,.255,.145],[.73,.73,.70],[.82,.78,.67],[.31,.37,.43],
            [.55,.59,.63],[.77,.86,.85],[.17,.19,.22],[.27,.17,.13]]))
        self.asset_hashes = {}
        self._register_assets(assets)

    @property
    def device(self):
        return self.x.device

    def _register_assets(self, assets):
        names = ['inverse_field','mask','normals','height','shading']
        if assets is None:
            for name in names:
                self.register_buffer('asset_'+name,torch.empty(0))
            return
        field = torch.as_tensor(assets.inverse_field,dtype=torch.float32)
        if field.ndim != 4 or field.shape[-1] != 2 or min(field.shape[1:3]) < 8 or field.shape[0] < 1:
            raise ValueError('Asset inverse_field must be [K,H,W,2] with H/W>=8')
        k,h,w,_ = field.shape
        defaults = {'mask':torch.ones(k,1,h,w), 'height':torch.zeros(k,1,h,w),
                    'shading':torch.ones(k,1,h,w),
                    'normals':torch.tensor([0.,0.,1.])[None,:,None,None].expand(k,3,h,w)}
        for name in names:
            value = field if name == 'inverse_field' else getattr(assets,name)
            value = defaults[name] if value is None else torch.as_tensor(value,dtype=torch.float32)
            expected = (k,h,w,2) if name == 'inverse_field' else (k,3 if name == 'normals' else 1,h,w)
            if tuple(value.shape) != expected or not torch.isfinite(value).all():
                raise ValueError(f'Invalid asset {name}')
            if name == 'mask' and (value.min() < 0 or value.max() > 1):
                raise ValueError('Asset visibility must be in [0,1]')
            if name in {'height','shading'} and value.min() < 0:
                raise ValueError('Asset height and shading must be nonnegative')
            if name == 'normals':
                mask = defaults['mask'] if assets.mask is None else torch.as_tensor(assets.mask,dtype=torch.float32)
                if ((value.square().sum(1,keepdim=True)<1e-6)&(mask>0)).any():
                    raise ValueError('Visible asset normals must be nonzero')
                value = F.normalize(value,dim=1)
            value = value.detach().clone().contiguous()
            self.register_buffer('asset_'+name,value)
            self.asset_hashes[name] = hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()

    def sample_parameters(self, b, intensity, geometry, *, generator=None, surface_mode='mixed'):
        rand = lambda *shape: torch.rand(shape or (b,),device=self.device,generator=generator)
        integer = lambda n: torch.randint(n,(b,),device=self.device,generator=generator)
        mode = integer(len(SURFACES)) if surface_mode == 'mixed' else torch.full(
            (b,), SURFACES.index(surface_mode) if surface_mode != 'off' else 0,
            device=self.device,dtype=torch.long)
        parameters = {'surface_mode':mode, 'surface_strength':(.25+rand()*.75)*geometry,
                'surface_axis':rand()*math.pi, 'surface_phase':rand()*math.tau,
                'surface_frequency':2+rand()*3,
                'relief_view':(rand(b,2)*2-1)*.52*geometry[:,None],
                'fold_angle':rand(b,8)*math.tau, 'fold_position':(rand(b,8)*2-1)*.85,
                'fold_amplitude':(.012+rand(b,8)*.048)*geometry[:,None],
                'fold_width':.020+rand(b,8)*.035,
                'fold_span':.12+rand(b,8)*.23,
                'fold_length':.45+rand(b,8)*.95,
                'fold_sign':torch.where(rand(b,8)<.5,-1.,1.),
                'crease_count':1+integer(8),
                'paper_gap':(.0006+rand()*.0012)*geometry,
                'corner_position':torch.where(rand(b,2)<.5,-1.,1.),
                'corner_size':.12+rand()*.27,
                'paper_palette':integer(len(self.paper_colors)),
                'paper_color':torch.ones(b,3,device=self.device),
                'background_type':integer(len(BACKGROUNDS)),
                'background_angle':rand()*math.tau, 'background_scale':.6+rand()*1.1,
                'background_tint':.90+rand(b,3)*.20,
                'light_direction':self._sample_light_directions(b, generator),
                'light_color':self._sample_light_colors(b, generator),
                'light_energy':(.18+rand(b,1)*.32)*intensity[:,None]
                               * torch.tensor([1.,.20,.08],device=self.device)[None],
                'ambient_light':1-intensity*(.27+rand()*.15),
                'specular_strength':rand()*intensity*.13,
                'roughness':.45+rand()*.50,
                'flash_strength':rand()*intensity*.20, 'flash_position':(rand(b,2)*2-1)*.6,
                'ambient_occlusion':rand()*geometry*.30,
                'cast_shadow_strength':(.15+rand()*.38)*geometry,
                'cast_shadow_softness':.004+rand()*.035,
                'object_shadow':integer(len(SHADOWS)),
                'object_shadow_strength':rand()*geometry*.58,
                'object_shadow_position':(rand(b,2)*2-1)*.8,
                'object_shadow_angle':rand()*math.tau,
                'object_shadow_size':.20+rand(b,2)*.70,
                'object_shadow_softness':.007+rand()*.11,
                'tear_active':(rand()<geometry*.12).float(),
                'tear_side':integer(4), 'tear_position':(rand()*2-1)*.70,
                'tear_size':.015+rand()*.10,
                'asset_active':(rand()<.65).float()* (geometry>0) * float(len(self.asset_mask)>0),
                'asset_index':integer(max(1,len(self.asset_mask)))}
        # Additional damage is opt-in. Reuse independent surface samples as
        # seeds, without consuming RNG state or changing zero-damage captures.
        phase = parameters['fold_angle'][:, :3]
        positions = parameters['fold_position'][:, :3]
        zeros = torch.zeros(b, device=self.device)
        parameters.update(
            tear_raggedness=torch.full_like(zeros, .65),
            tear_count=torch.ones(b, device=self.device, dtype=torch.long),
            edge_wear_strength=zeros.clone(),
            edge_wear_depth=torch.full_like(zeros, .008),
            hole_active=zeros[:, None].expand(-1, 3).clone(),
            hole_center=torch.stack((positions * .65, phase.cos() * .55), -1),
            hole_radius=torch.stack((.025 + phase.sin().abs() * .025,
                                     .025 + phase.cos().abs() * .030), -1),
            hole_angle=phase.clone(),
            cut_active=zeros[:, None].expand(-1, 3).clone(),
            cut_center=torch.stack((positions * .70, phase.sin() * .65), -1),
            cut_length=.12 + phase.cos().abs() * .18,
            cut_width=.003 + phase.sin().abs() * .003,
            cut_angle=phase.clone(),
            missing_corner_active=zeros.clone(),
            missing_corner_index=parameters['tear_side'].clone(),
            missing_corner_size=torch.stack((.12 + positions[:, 0].abs() * .12,
                                              .12 + positions[:, 1].abs() * .12), -1))
        return parameters

    def _sample_light_directions(self, b, generator):
        """One key and two weak room bounces, rather than three equal suns."""
        xy = torch.rand(b, 1, 2, device=self.device, generator=generator) * 2 - 1
        z = .65 + torch.rand(b, 1, 1, device=self.device, generator=generator)
        key = torch.cat((xy, z), -1)
        fill = torch.cat((-xy * .40, torch.ones_like(z) * 1.4), -1)
        overhead = torch.cat((xy * .12, torch.ones_like(z) * 2), -1)
        return torch.cat((key, fill, overhead), 1)

    def _sample_light_colors(self, b, generator):
        # Correlated warm/cool shifts avoid random RGB theatrical lighting.
        warmth = (torch.rand(b, 1, 1, device=self.device, generator=generator) * 2 - 1) * .045
        channels = torch.tensor([1., .12, -1.], device=self.device)[None, None]
        return (1 + warmth * channels).expand(-1, 3, -1)

    def surface(self, p, generator):
        """Material mesh, normals and observable metric error at scene scale."""
        mesh, normals, metric = self.surface_model(p, generator)
        mesh = torch.cat((mesh[:, :2], mesh[:, 2:] + p['paper_gap'][:, None, None, None]), 1)
        return mesh, normals, metric

    def _light_visibility(self, height, directions):
        b, lights, _ = directions.shape
        h, w = height.shape[-2:]
        visibility = torch.ones(b, lights, 1, h, w, device=self.device)
        if not self.shadow_samples:
            return visibility
        source = height[:, None].expand(-1, lights, -1, -1, -1).reshape(b * lights, 1, h, w)
        positions = torch.stack((self.x[:, 0].expand(b, -1, -1), self.y[:, 0].expand(b, -1, -1)), -1)
        # Bound temporary memory while processing all lights together. Eight
        # ray samples require one grid_sample call instead of 24 serial calls.
        for offset in range(0, self.shadow_samples, 8):
            steps = torch.arange(offset, min(offset + 8, self.shadow_samples), device=self.device)
            distances = .015 + (steps + 1) * (.30 / self.shadow_samples)
            rays = positions[:, None, None] + directions[:, :, None, None, None, :2] * distances[None, None, :, None, None, None]
            field = torch.stack((rays[..., 0], rays[..., 1] / self.aspect), -1)
            samples = F.grid_sample(source, field.reshape(b * lights, -1, w, 2), padding_mode='border', align_corners=False)
            samples = samples.reshape(b, lights, len(steps), 1, h, w)
            delta = samples - height[:, None, None] - directions[:, :, 2, None, None, None, None] * distances[None, None, :, None, None, None]
            inside = (field.abs() <= 1).all(-1)[:, :, :, None]
            # A nonintersecting ray must remain fully visible, including at
            # grazing angles. A symmetric sigmoid falsely shadows flat paper.
            blocker = (1 - torch.exp(-(delta - .0015).clamp_min(0) / .0025)) * inside
            visibility = torch.minimum(visibility, (1 - blocker).amin(2))
        return visibility

    def illumination(self, height, normals, p, *, procedural=True):
        """Energy-normalized matte shading using one coherent surface normal."""
        light = F.normalize(p['light_direction'], dim=-1)
        n = normals[:, None]
        directions = light[:, :, :, None, None]
        lambert = (n * directions).sum(2, keepdim=True).clamp_min(0)
        visibility = self._light_visibility(height, light) if procedural else torch.ones_like(lambert)
        view = torch.cat((torch.zeros_like(light[..., :2]), torch.ones_like(light[..., 2:])), -1)
        halfway = F.normalize(light + view, dim=-1)
        nh = (n * halfway[:, :, :, None, None]).sum(2, keepdim=True).clamp(0, 1)
        nv = normals[:, 2:3, None].transpose(1, 2).clamp_min(.05)
        alpha2 = p['roughness'].clamp(.45, 1).pow(4)[:, None, None, None, None]
        distribution = alpha2 / (math.pi * (nh.square() * (alpha2 - 1) + 1).square().clamp_min(1e-4))
        vh = halfway[:, :, 2, None, None, None].clamp(0, 1)
        fresnel = .04 + .96 * (1 - vh).pow(5)
        specular = (distribution * fresnel / (4 * nv)).clamp(max=1) * p['specular_strength'][:, None, None, None, None]
        energy = p['light_energy'][:, :, None, None, None]
        colors = p['light_color'][:, :, :, None, None]
        diffuse = ((lambert + specular) * visibility * energy * colors).sum(1)
        ambient = p['ambient_light'][:, None, None, None]
        result = ambient * (.94 + .06 * normals[:, 2:3].clamp(0, 1)) + diffuse
        flat = ambient + (light[:, :, 2].clamp_min(0)[:, :, None, None, None] * energy * colors).sum(1)
        result = result / flat.clamp_min(.1)
        ao = (F.avg_pool2d(height, 7, 1, 3, count_include_pad=False) - height).clamp_min(0)
        result = result * (1 - (ao * 16).clamp(0, 1) * p['ambient_occlusion'][:, None, None, None])
        # Tiny crease wear follows material curvature, rather than a painted line.
        if procedural:
            hx, hy = derivatives(height, self.aspect)
            hxx, _ = derivatives(hx, self.aspect)
            _, hyy = derivatives(hy, self.aspect)
            wear = ((hxx.abs() + hyy.abs()) * .025).clamp(0, 1)
            result = result * (1 - wear * p['crease_strength'][:, None, None, None])
        xy = self.surface_model.material
        if height.shape[-2:] != xy.shape[-2:]:
            xy = F.interpolate(xy, size=height.shape[-2:], mode='bilinear', align_corners=False)
        flash = torch.exp(-((xy[:, :1] - p['flash_position'][:, 0, None, None, None]).square()
                            + (xy[:, 1:] - p['flash_position'][:, 1, None, None, None]).square()) / .40)
        result = result + flash * p['flash_strength'][:, None, None, None]
        return result.clamp(.18, 1.35)

    def deform(self, base_field, p, generator, *, diagnostics=True):
        """Invert material-space projection, with coarse solve and pixel polish.

        The scene uses the same pixel-center convention as align_corners=False.
        A final residual and positive Jacobian gate visibility. Dense mode is
        available as an independent numerical reference for the coarse solver.
        """
        mesh, normals, metric = self.surface(p, generator)
        height = mesh[:, 2:]
        projection = mesh[:, :2] + height * p['relief_view'][:, :, None, None]
        du, dv = derivatives(projection, self.aspect)
        displacement = projection - self.surface_model.material
        jacobian = torch.cat((du[:, :1], dv[:, :1], du[:, 1:], dv[:, 1:]), 1)
        maps = torch.cat((displacement, jacobian), 1)
        determinant = du[:, :1] * dv[:, 1:] - dv[:, :1] * du[:, 1:]
        shading = self._illumination_kernel(height, normals, p)
        wanted = torch.stack((base_field[..., 0], base_field[..., 1] * self.aspect), -1)
        size = mesh.shape[-2:]
        if self.geometry_solver == 'coarse_to_fine':
            wanted_small = F.interpolate(wanted.permute(0, 3, 1, 2), size=size, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            uv = wanted_small.clone()
            for _ in range(max(2, self.inverse_iterations - 1)):
                field = torch.stack((uv[..., 0], uv[..., 1] / self.aspect), -1)
                values = F.grid_sample(maps, field, padding_mode='border', align_corners=False)
                uv = self._newton_step(uv, wanted_small, values)
            uv = F.interpolate(uv.permute(0, 3, 1, 2), size=base_field.shape[1:3], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            refinements = 2
        else:
            uv, refinements = wanted.clone(), self.inverse_iterations
        for _ in range(refinements):
            field = torch.stack((uv[..., 0], uv[..., 1] / self.aspect), -1)
            values = F.grid_sample(maps, field, padding_mode='border', align_corners=False)
            uv = self._newton_step(uv, wanted, values)
        field = torch.stack((uv[..., 0], uv[..., 1] / self.aspect), -1)
        final_maps = torch.cat((displacement, determinant, height, shading), 1)
        if diagnostics:
            final_maps = torch.cat((final_maps, normals, metric), 1)
        values = F.grid_sample(final_maps, field, padding_mode='border', align_corners=False)
        residual = (uv + values[:, :2].permute(0, 2, 3, 1) - wanted).square().sum(-1).sqrt()
        valid = ((values[:, 2:3] > .12) & (residual[:, None] < 2 / self.width)).float()
        height_image, light_image = values[:, 3:4], values[:, 4:7]
        normal_image = F.normalize(values[:, 7:10], dim=1) if diagnostics else None
        metric_image = values[:, 10:13] if diagnostics else None
        asset_id = torch.full((len(p['asset_index']),), -1, device=self.device, dtype=torch.long)
        if len(self.asset_mask):
            ids = p['asset_index'].long()
            source = torch.cat((self.asset_inverse_field[ids].permute(0, 3, 1, 2), self.asset_height[ids],
                                self.asset_normals[ids], self.asset_shading[ids]), 1)
            assets = F.grid_sample(source, base_field, padding_mode='border', align_corners=False)
            # Visibility uses zero padding. Border padding for UV/normals must
            # never replicate a visible bank edge beyond the source camera.
            asset_mask = F.grid_sample(self.asset_mask[ids], base_field, align_corners=False)
            asset_field, asset_height = assets[:, :2].permute(0, 2, 3, 1), assets[:, 2:3]
            asset_normals = F.normalize(assets[:, 3:6], dim=1)
            asset_light = self._illumination_kernel(asset_height, asset_normals, p, procedural=False) * assets[:, 6:7]
            active = p['asset_active'][:, None, None, None].bool()
            field = torch.where(active.permute(0, 2, 3, 1), asset_field, field)
            valid = torch.where(active, asset_mask, valid)
            light_image = torch.where(active, asset_light, light_image)
            height_image = torch.where(active, asset_height, height_image)
            if diagnostics:
                normal_image = torch.where(active, asset_normals, normal_image)
                metric_image = torch.where(active, torch.zeros_like(metric_image), metric_image)
            asset_id = torch.where(p['asset_active'].bool(), ids, asset_id)
            residual = torch.where(p['asset_active'][:, None, None].bool(), torch.zeros_like(residual), residual)
        result = {'field': field, 'solver_valid': valid, 'height': height_image,
                  'illumination': light_image, 'asset_id': asset_id, 'projection_residual': residual[:, None]}
        if diagnostics:
            result.update(normals=normal_image, metric_strain=metric_image)
        return result

    def background(self,p,generator):
        """Subtle, nonperiodic table textures generated above the scene scale."""
        b = len(p['background_type'])
        x, y = self.background_xy[:, :1].expand(b, -1, -1, -1), self.background_xy[:, 1:].expand(b, -1, -1, -1)
        angle = p['background_angle'][:, None, None, None]
        u, v = x * angle.cos() + y * angle.sin(), -x * angle.sin() + y * angle.cos()
        scale = p['background_scale'][:, None, None, None]
        noise = []
        for h, w in [(9, 13), (27, 39), (63, 87)]:
            value = torch.randn(b, 1, h, w, device=self.device, generator=generator)
            noise.append(F.interpolate(value, size=x.shape[-2:], mode='bilinear', align_corners=False))
        coarse, middle, fine = noise
        # Anisotropic *stochastic* fibres. The old sinusoidal wood looked
        # like identical parallel waves at every scale and across every seed.
        # Rotate a random elongated field into material coordinates instead.
        grain = []
        field = torch.stack((u[:, 0] * .70 / scale[:, 0],
                             v[:, 0] * .70 / (self.aspect * scale[:, 0])), -1)
        for gh, gw in [(37, 5), (131, 17)]:
            value = torch.randn(b, 1, gh, gw, device=self.device, generator=generator)
            grain.append(F.grid_sample(value, field, padding_mode='reflection', align_corners=False))
        broad_grain, small_grain = grain
        wood = coarse * .009 + middle * .004 + broad_grain * .014 + small_grain * .005
        # Irregular nonperiodic ridge bands are only a fallback material.
        # Real texture banks remain preferable for identifiable substrates.
        turbulence = coarse * .55 + middle * .22 + fine * .07
        veins = torch.exp(-((turbulence - .16) / .035).square())
        marble = -veins * .033 + coarse * .006 + fine * .002
        laminate = wood * .21 + middle * .002
        fabric = fine * .010 + middle * .004 + small_grain * .002
        metal = coarse * .008 + small_grain * .004
        clinical = coarse * .0025 + fine * .0015
        plain = coarse * .002 + fine * .001
        leather = .010 * torch.tanh(fine * 2) + middle * .006 + coarse * .003
        bank = torch.cat((wood, marble, laminate, fabric, metal, clinical, plain, leather), 1)
        texture = bank.gather(1, p['background_type'][:, None, None, None].expand(-1, 1, *x.shape[-2:]))
        color = self.table_colors[p['background_type'].long()][:, :, None, None]
        image = (color + texture) * p['background_tint'][:, :, None, None]
        return F.interpolate(image.clamp(0,1), size=(self.height,self.width), mode='bilinear', align_corners=False).clamp(0,1)

    @staticmethod
    def _fracture_profile(position, phase):
        """Bounded multiscale edge variation in material half-width units.

        Incommensurate spatial frequencies avoid the repeated sawtooth edge of
        a single sine. This is a geometric damage model, not fibre mechanics.
        """
        result = torch.zeros_like(position)
        for amplitude, frequency, shift in [(.53, 31., .0), (.27, 79., 1.7),
                                             (.13, 193., 3.1), (.07, 463., 5.4)]:
            result = result + amplitude * torch.sin(position * frequency + phase * (1 + shift) + shift)
        return result

    def paper_mask(self, field, p):
        """Binary surviving material, shared by RGB, targets and cast shadows.

        All damage follows the inverse material coordinates rather than image
        axes. Centers use normalized sheet UV, while radii, lengths, widths and
        depths use the same physical half-page-width unit in both directions.
        Thus .01 is 1.5 mm on a 300 mm wide sheet. The zero-damage result is
        exactly one. Static damage_detail='edge' evaluates only one edge tear
        for faster training, ignoring the advanced rich-damage parameters.
        Source-frame coverage remains the renderer's separate
        responsibility and is never multiplied here a second time.
        """
        x, y = field[..., 0], field[..., 1] * self.aspect
        phase = p['surface_phase'][:, None, None]
        raggedness = p['tear_raggedness'][:, None, None]
        missing = torch.zeros_like(x, dtype=torch.bool)

        # Shared material fibre variation makes nearby edge defects coherent
        # and avoids evaluating a separate full-frame noise field per defect.
        if self.damage_detail == 'rich':
            rough_x = self._fracture_profile(x, phase)
            rough_y = self._fracture_profile(y, phase + 2.3)

        # Up to four differently sized open edge tears. The first preserves
        # explicit tear_side / tear_position control from previous releases.
        for index in range(4 if self.damage_detail == 'rich' else 1):
            side = (p['tear_side'][:, None, None] + index) % 4
            inward = torch.where(side < 2, 1 - x.abs(), self.aspect - y.abs())
            along = torch.where(side < 2, y, x)
            along_scale = torch.where(side < 2, self.aspect, 1.)
            center = (p['tear_position'][:, None, None] if index == 0
                      else p['fold_position'][:, index, None, None]) * along_scale
            size = p['tear_size'][:, None, None] * (1 - index * .13)
            # Off-center flank slopes and a rough advancing tear front replace
            # the old symmetric triangular bite with a fixed sine decoration.
            local = (along - center) / size.clamp_min(1e-6)
            skew = .22 * torch.sin(phase + index * 1.9) * raggedness
            flank = torch.where(local < 0, 1 + skew, 1 - skew)
            envelope = (1 - local.abs() / flank.clamp_min(.5)).clamp_min(0).pow(.72)
            rough = (torch.where(side < 2, rough_y, rough_x)
                     if self.damage_detail == 'rich' else self._fracture_profile(along, phase))
            depth = size * envelope * (1 + .28 * rough * raggedness)
            correct_edge = torch.where(side == 0, x > 0,
                           torch.where(side == 1, x < 0,
                           torch.where(side == 2, y > 0, y < 0)))
            active = p['tear_active'][:, None, None].bool() & (p['tear_count'][:, None, None] > index)
            missing = missing | ((inward < depth) & (envelope > 0) & correct_edge & active)

        if self.damage_detail == 'edge':
            return (~missing)[:, None].to(field.dtype)

        # Distributed shallow abrasion of the outer edge. Its largest possible
        # depth is explicitly bounded rather than removing an arbitrary page.
        edge_strength = p['edge_wear_strength'][:, None, None]
        edge_depth = p['edge_wear_depth'][:, None, None]
        for axis, sign in [(0, -1.), (0, 1.), (1, -1.), (1, 1.)]:
            inward = 1 - sign * x if axis == 0 else self.aspect - sign * y
            along = y if axis == 0 else x
            noise = (rough_y if axis == 0 else rough_x) * sign
            wear = edge_depth * edge_strength * (.5 + .5 * noise).clamp(0, 1).pow(2)
            missing = missing | ((inward >= 0) & (inward < wear))

        # Internal irregular punctures and finite open slits are distinct from
        # stains or shadows. They remove physical substrate and all trace GT.
        for index in range(3):
            cx = p['hole_center'][:, index, 0, None, None]
            cy = p['hole_center'][:, index, 1, None, None] * self.aspect
            angle = p['hole_angle'][:, index, None, None]
            dx, dy = x - cx, y - cy
            u, v = dx * angle.cos() + dy * angle.sin(), -dx * angle.sin() + dy * angle.cos()
            u = u / p['hole_radius'][:, index, 0, None, None].clamp_min(1e-6)
            v = v / p['hole_radius'][:, index, 1, None, None].clamp_min(1e-6)
            theta = torch.atan2(v, u)
            boundary = (1 + raggedness * (.15 * torch.sin(theta * 3 + phase)
                       + .10 * torch.sin(theta * 7 - phase * 1.3)
                       + .05 * torch.sin(theta * 13 + phase * 2.1)))
            missing = missing | ((u.square() + v.square() < boundary.square())
                                 & p['hole_active'][:, index, None, None].bool())

            cx = p['cut_center'][:, index, 0, None, None]
            cy = p['cut_center'][:, index, 1, None, None] * self.aspect
            angle = p['cut_angle'][:, index, None, None]
            dx, dy = x - cx, y - cy
            u, v = dx * angle.cos() + dy * angle.sin(), -dx * angle.sin() + dy * angle.cos()
            half_length = p['cut_length'][:, index, None, None] * .5
            nearest = u.clamp(-half_length, half_length)
            width = p['cut_width'][:, index, None, None]
            bend = (torch.sin(nearest * 79 + phase + index * 1.1) * .65
                    + torch.sin(nearest * 193 - phase * 1.7) * .35) * width * raggedness * .40
            # A narrow capsule gives an actual kerf, with a mildly wandering
            # centerline. It does not create a fictitious dark ink stripe.
            distance2 = (u - nearest).square() + (v - bend).square()
            missing = missing | ((distance2 < (width * .5).square())
                                 & p['cut_active'][:, index, None, None].bool())

        # A removed corner is an open material boundary, unlike a curled corner
        # which remains present in the surface mesh. Order TL, TR, BR, BL.
        corner = p['missing_corner_index'][:, None, None]
        sx = torch.where((corner == 1) | (corner == 2), 1., -1.)
        sy = torch.where(corner >= 2, 1., -1.)
        dx, dy = 1 - sx * x, self.aspect - sy * y
        span_x = p['missing_corner_size'][:, 0, None, None].clamp_min(1e-6)
        span_y = p['missing_corner_size'][:, 1, None, None].clamp_min(1e-6)
        front = 1 + .14 * raggedness * self._fracture_profile(dx - dy, phase + 4.2)
        corner_missing = (dx / span_x + dy / span_y < front)
        missing = missing | (corner_missing & p['missing_corner_active'][:, None, None].bool())
        return (~missing)[:, None].to(field.dtype)

    @staticmethod
    def _project_shadow(mask, height, offsets):
        """Forward-splat caster pixels onto a plane using their own depth.

        Offsets are displacement in pixels per unit surface height. Unlike an
        inverse shift using receiver depth, this preserves the correct shadow
        for a lifted edge or an isolated elevated sheet. Four-corner splats
        preserve subpixel translations and do not wrap around image edges.
        """
        b, lights, _ = offsets.shape
        h, w = mask.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h, device=mask.device),
                                torch.arange(w, device=mask.device), indexing='ij')
        x = xx[None, None] - height[:, 0, None] * offsets[:, :, 0, None, None]
        y = yy[None, None] - height[:, 0, None] * offsets[:, :, 1, None, None]
        x0, y0 = x.floor(), y.floor()
        fx, fy = x - x0, y - y0
        result = torch.zeros(b, lights, h*w, device=mask.device)
        for dx, dy, weight in [(0,0,(1-fx)*(1-fy)), (1,0,fx*(1-fy)),
                               (0,1,(1-fx)*fy), (1,1,fx*fy)]:
            ix, iy = (x0+dx).long(), (y0+dy).long()
            inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            index = iy.clamp(0,h-1)*w + ix.clamp(0,w-1)
            values = mask[:, 0, None] * weight * inside
            result.scatter_add_(2, index.flatten(2), values.flatten(2))
        return result.reshape(b, lights, h, w).clamp(0,1)

    def cast_shadow(self,paper_mask,scene,p):
        """Directional area-light shadows and depth-gated contact occlusion.

        Projection is a weak-perspective tabletop approximation. Source height
        comes from the exact paper map shared by image and target geometry.
        It is not a full 3D visibility or interreflection solver.
        """
        b, _, h, w = paper_mask.shape
        size = self.background_xy.shape[-2:]
        mask = F.interpolate(paper_mask, size=size, mode='area')
        height = F.interpolate(scene['height'], size=size, mode='bilinear', align_corners=False)
        light = F.normalize(p['light_direction'], dim=-1)
        c, s = p['angle'].cos()[:, None], p['angle'].sin()[:, None]
        lx = light[:, :, 0] * c - light[:, :, 1] * s
        ly = light[:, :, 0] * s + light[:, :, 1] * c
        offsets = torch.stack((lx, ly), -1) / light[:, :, 2:].clamp_min(.10)
        offsets = offsets * p['zoom'][:, None, None] * (size[1] / 2)
        # Deterministic quadrature over a small angular emitter. Separation
        # grows linearly with distance from the receiving plane, so an edge
        # touching the table stays sharper than a lifted edge of the same page.
        radius = p['cast_shadow_softness'][:, None, None] * (size[1] / 2)
        soft = torch.zeros(b, 3, *size, device=self.device)
        for du, dv in [(0.,0.), (1.,0.), (-1.,0.), (0.,1.), (0.,-1.)]:
            jitter = torch.tensor([du,dv], device=self.device)[None, None] * radius
            soft = soft + self._project_shadow(mask, height, offsets+jitter) / 5
        soft = F.avg_pool2d(soft,3,1,1,count_include_pad=False)
        weights = p['light_energy'] / p['light_energy'].sum(1, keepdim=True).clamp_min(.001)
        cast = (soft * weights[:, :, None, None]).sum(1,keepdim=True)
        cast = F.interpolate(cast, size=(h, w), mode='bilinear', align_corners=False)
        # A page at 10 mm above the receiver cannot retain an attached black
        # outline. Only genuinely near-plane portions contribute this term.
        proximity = torch.exp(-scene['height'].clamp_min(0) / .003) * paper_mask
        contact = F.avg_pool2d(F.max_pool2d(proximity,3,1,1),3,1,1) * .32
        result = torch.maximum(cast, contact) * p['cast_shadow_strength'][:, None, None, None]
        return result.clamp(0,1) * (1 - paper_mask)

    def object_shadow(self,p):
        b = len(p['object_shadow'])
        x,y = self.x.expand(b,-1,-1,-1),self.y.expand(b,-1,-1,-1)
        x=x-p['object_shadow_position'][:,0,None,None,None]
        y=y-p['object_shadow_position'][:,1,None,None,None]
        angle=p['object_shadow_angle'][:,None,None,None]
        u,v=x*angle.cos()+y*angle.sin(),-x*angle.sin()+y*angle.cos()
        sx,sy=p['object_shadow_size'][:,0,None,None,None],p['object_shadow_size'][:,1,None,None,None]
        softness=p['object_shadow_softness'][:,None,None,None]
        ellipse=((u/sx).square()+(v/sy).square()).sqrt()-1
        phone=torch.maximum(u.abs()-sx*.45,v.abs()-sy*.70)
        palm=((u/(sx*.5)).square()+(v/(sy*.4)).square()).sqrt()-1
        hand=palm*min(self.aspect,1)*.25
        for index in range(4):
            center=(index-1.5)*sx*.24
            finger=torch.maximum((u-center).abs()-sx*.105,(v+sy*.45).abs()-sy*.47)
            hand=torch.minimum(hand,finger)
        blinds=torch.sin(u/(sx*.3)*math.pi)*.10
        window=torch.minimum((u.abs()-sx*.07),(v.abs()-sy*.07))
        distances=[torch.full_like(u,10),u,phone,hand,blinds,window,ellipse*.25]
        shadow=torch.zeros_like(u)
        kind=p['object_shadow'][:,None,None,None]
        for index,distance in enumerate(distances):
            shadow+=torch.sigmoid(-distance/softness)*(kind==index)
        shadow*=p['object_shadow_strength'][:,None,None,None]
        return F.interpolate(shadow,size=(self.height,self.width),mode='bilinear',align_corners=False)
