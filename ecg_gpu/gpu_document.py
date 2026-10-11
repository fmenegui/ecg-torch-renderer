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
                 shadow_samples=8, assets=None, geometry_solver='coarse_to_fine', compile_scene=False):
        super().__init__()
        self.height, self.width, self.aspect = height, width, height / width
        self.resolution = resolution
        self.inverse_iterations, self.shadow_samples = inverse_iterations, shadow_samples
        if geometry_solver not in {'coarse_to_fine', 'dense'}:
            raise ValueError('geometry_solver must be coarse_to_fine or dense')
        if not isinstance(compile_scene, bool):
            raise ValueError('compile_scene must be a boolean')
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
            [.985,.980,.970],[.995,.910,.905],[.970,.950,.840],[.970,.915,.710],
            [.865,.940,.980],[.865,.955,.880],[.900,.905,.895],[.875,.800,.655]]))
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
        return {'surface_mode':mode, 'surface_strength':(.25+rand()*.75)*geometry,
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
                'light_direction':torch.cat([(rand(b,3,2)*2-1)*1.1,.65+rand(b,3,1)],-1),
                'light_color':.86+rand(b,3,3)*.14,
                'light_energy':(.18+rand(b,3)*.32)*intensity[:,None]/2,
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
        grain_coordinate = v * scale + coarse * .014 + middle * .005 + .022 * torch.sin(u * 3.8)
        wood = .022 * torch.sin(grain_coordinate * 68) + .008 * torch.sin(grain_coordinate * 153 + middle * .18) + coarse * .010 + fine * .004
        veins = torch.sin((u + v * .31 + coarse * .16) * scale * 7.5)
        marble = -.068 * torch.exp(-(veins / .095).square()) + middle * .006 + fine * .003
        laminate = wood * .28 + middle * .003
        fabric = fine * .012 + middle * .004 + .004 * (torch.sin(u * 82) + torch.sin(v * 86))
        metal = coarse * .011 + fine * .005 + .004 * torch.sin(v * 108 + middle * .4)
        clinical = coarse * .0025 + fine * .0015
        plain = coarse * .002 + fine * .001
        leather = .012 * torch.tanh(fine * 2) + middle * .007 + coarse * .004
        bank = torch.cat((wood, marble, laminate, fabric, metal, clinical, plain, leather), 1)
        texture = bank.gather(1, p['background_type'][:, None, None, None].expand(-1, 1, *x.shape[-2:]))
        color = self.table_colors[p['background_type'].long()][:, :, None, None]
        image = (color + texture) * p['background_tint'][:, :, None, None]
        return F.interpolate(image.clamp(0,1), size=(self.height,self.width), mode='bilinear', align_corners=False).clamp(0,1)

    def paper_mask(self,field,p):
        # Ragged tears remove paper and its ground-truth visibility together.
        x,y = field[...,0],field[...,1]
        sides = p['tear_side'][:,None,None]
        inward = torch.where(sides<2,1-x.abs(),1-y.abs())
        along = torch.where(sides<2,y,x)
        edge_side = torch.where(sides==0,x>0,torch.where(sides==1,x<0,torch.where(sides==2,y>0,y<0)))
        notch = p['tear_size'][:,None,None]*(1-(along-p['tear_position'][:,None,None]).abs()/p['tear_size'][:,None,None]).clamp_min(0)
        notch *= .82+.18*torch.sin(along*360).abs()
        tear = (inward<notch)&edge_side&p['tear_active'][:,None,None].bool()
        return (~tear)[:,None].float()

    def cast_shadow(self,paper_mask,scene,p):
        """Local-height cast shadows and a narrow sheet/table contact shadow."""
        b, _, h, w = paper_mask.shape
        size = self.background_xy.shape[-2:]
        mask = F.interpolate(paper_mask, size=size, mode='area')
        height = F.interpolate(scene['height'], size=size, mode='bilinear', align_corners=False)
        light = F.normalize(p['light_direction'], dim=-1)
        c, s = p['angle'].cos()[:, None], p['angle'].sin()[:, None]
        lx = light[:, :, 0] * c - light[:, :, 1] * s
        ly = light[:, :, 0] * s + light[:, :, 1] * c
        offsets = torch.stack((lx, ly / self.aspect), -1) / light[:, :, 2:].clamp_min(.25)
        gap = height[:, None] * p['zoom'][:, None, None, None, None]
        base = torch.stack((self.background_xy[:, 0], self.background_xy[:, 1] / self.aspect), -1)
        field = base[:, None] + offsets[:, :, None, None] * gap.permute(0, 1, 3, 4, 2)
        source = mask[:, None].expand(-1, 3, -1, -1, -1).reshape(b * 3, 1, *size)
        shifted = F.grid_sample(source, field.reshape(b * 3, *size, 2), align_corners=False)
        near = F.avg_pool2d(shifted, 3, 1, 1)
        wide = F.avg_pool2d(shifted, 9, 1, 4)
        softness = (p['cast_shadow_softness'] * self.resolution / 4).clamp(0, 1)
        blend = softness[:, None].expand(-1, 3).reshape(b * 3, 1, 1, 1)
        soft = (near * (1 - blend) + wide * blend).reshape(b, 3, 1, *size)
        weights = p['light_energy'] / p['light_energy'].sum(1, keepdim=True).clamp_min(.001)
        cast = (soft * weights[:, :, None, None, None]).sum(1)
        cast = F.interpolate(cast, size=(h, w), mode='bilinear', align_corners=False)
        contact = F.avg_pool2d(F.max_pool2d(paper_mask, 3, 1, 1), 3, 1, 1) * .45
        result = torch.maximum(cast, contact) * p['cast_shadow_strength'][:, None, None, None]
        return result * (1 - paper_mask)

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
