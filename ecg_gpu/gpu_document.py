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
                 shadow_samples=8, assets=None):
        super().__init__()
        self.height, self.width, self.aspect = height, width, height / width
        self.resolution = resolution
        self.inverse_iterations, self.shadow_samples = inverse_iterations, shadow_samples
        rows = max(24, round(resolution * self.aspect))
        y, x = torch.meshgrid(torch.linspace(-self.aspect,self.aspect,rows),
                              torch.linspace(-1,1,resolution), indexing='ij')
        self.register_buffer('x',x[None,None].clone())
        self.register_buffer('y',y[None,None].clone())
        self.register_buffer('paper_colors',torch.tensor([
            [.985,.980,.970],[.995,.910,.905],[.970,.950,.840],[.970,.915,.710],
            [.865,.940,.980],[.865,.955,.880],[.900,.905,.895],[.875,.800,.655]]))
        self.register_buffer('table_colors',torch.tensor([
            [.42,.255,.145],[.73,.73,.70],[.82,.78,.67],[.31,.37,.43],
            [.55,.59,.63],[.77,.86,.85],[.17,.19,.22],[.27,.17,.13]]))
        self.register_buffer('noise_levels',torch.tensor([.050,.021,.009,.0035])[None,:,None,None])
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
                'relief_view':(rand(b,2)*2-1)*.72*geometry[:,None],
                'fold_angle':rand(b,8)*math.tau, 'fold_position':(rand(b,8)*2-1)*.85,
                'fold_amplitude':(.006+rand(b,8)*.026)*geometry[:,None],
                'fold_width':.010+rand(b,8)*.050,
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
        b = len(p['surface_mode'])
        x,y = self.x.expand(b,-1,-1,-1), self.y.expand(b,-1,-1,-1)
        ca,sa = p['surface_axis'].cos()[:,None,None,None],p['surface_axis'].sin()[:,None,None,None]
        u,v = x*ca+y*sa,-x*sa+y*ca
        phase = p['surface_phase'][:,None,None,None]
        frequency = p['surface_frequency'][:,None,None,None]
        curl = .14*u.square()+.025*torch.sin(v*2+phase)
        fold = .14*(u-.15*torch.sin(phase)).abs()
        accordion = .027*torch.sqrt(torch.sin(u*frequency*math.pi+phase).square()+.012)
        wrinkles = .008*torch.sin(u*frequency*math.pi*3+phase)*torch.cos(v*3+phase)
        rolled = .22*torch.relu(u-.10).square()
        corner_distance = ((x-p['corner_position'][:,0,None,None,None]).square()
                           +(y/self.aspect-p['corner_position'][:,1,None,None,None]).square()).sqrt()
        corner = .28*torch.relu(1-corner_distance/p['corner_size'][:,None,None,None]).square()
        mode = p['surface_mode'][:,None,None,None]
        height = torch.zeros_like(x)
        for index,value in [(1,curl),(2,fold),(3,accordion),(5,wrinkles),(6,rolled),(7,corner)]:
            height += value*(mode==index)
        crumple = torch.zeros_like(height)
        for index,size in enumerate([4,9,21,37]):
            noise = torch.randn(b,1,max(3,round(size*self.aspect)),size,device=self.device,generator=generator)
            crumple += F.interpolate(noise,size=x.shape[-2:],mode='bicubic',align_corners=False)*self.noise_levels[:,index:index+1]
        height += crumple*((mode==4)+(mode==5)*.12)
        for index in range(8):
            ca = p['fold_angle'][:,index,None,None,None].cos()
            sa = p['fold_angle'][:,index,None,None,None].sin()
            distance = x*ca+y*sa-p['fold_position'][:,index,None,None,None]
            taper = torch.exp(-((x*sa-y*ca)/1.2).square())
            ridge = torch.exp(-(distance/p['fold_width'][:,index,None,None,None]).square())*taper
            height += ridge*p['fold_amplitude'][:,index,None,None,None]*((mode==4)+(mode==5)*.35)
        height *= p['surface_strength'][:,None,None,None]
        height = height-height.amin((-1,-2),keepdim=True)
        dx = (F.pad(height,(1,1,0,0),mode='replicate')[...,2:]-F.pad(height,(1,1,0,0),mode='replicate')[...,:-2])/(4/(height.shape[-1]-1))
        dy = (F.pad(height,(0,0,1,1),mode='replicate')[...,2:,:]-F.pad(height,(0,0,1,1),mode='replicate')[...,:-2,:])/(4*self.aspect/(height.shape[-2]-1))
        normals = F.normalize(torch.cat([-dx,-dy,torch.ones_like(dx)],1),dim=1)
        return torch.cat([height,dx,dy],1),normals

    def illumination(self, height, normals, p, *, procedural=True):
        b,_,h,w = height.shape
        result = p['ambient_light'][:,None,None,None].expand(b,3,h,w).clone()
        positions = torch.stack([self.x.expand(b,1,h,w)[:,0],self.y.expand(b,1,h,w)[:,0]],-1)
        ao = (F.avg_pool2d(height,7,1,3)-height).clamp_min(0)
        for index in range(3):
            light = F.normalize(p['light_direction'][:,index],dim=-1)
            lambert = (normals*light[:,:,None,None]).sum(1,keepdim=True).clamp_min(0)
            visibility = torch.ones_like(height)
            if procedural and self.shadow_samples:
                for step in range(self.shadow_samples):
                    distance = .015+(step+1)/self.shadow_samples*.42
                    ray = positions+light[:,:2,None,None].permute(0,2,3,1)*distance
                    field = torch.stack([ray[...,0],ray[...,1]/self.aspect],-1)
                    sampled = F.grid_sample(height,field,padding_mode='border',align_corners=False)
                    delta = sampled-height-light[:,2,None,None,None]*distance
                    blocker = torch.sigmoid((delta-.0015)/.0025)
                    inside = ((field.abs()<=1).all(-1))[:,None]
                    visibility = torch.minimum(visibility,1-blocker*inside)
            halfway = F.normalize(light+torch.tensor([0.,0.,1.],device=self.device),dim=-1)
            specular = (normals*halfway[:,:,None,None]).sum(1,keepdim=True).clamp_min(0)
            specular = specular.pow((8+40*(1-p['roughness']))[:,None,None,None])
            diffuse = lambert*p['light_energy'][:,index,None,None,None]
            result += (diffuse+specular*p['specular_strength'][:,None,None,None])*visibility*p['light_color'][:,index,:,None,None]
        result *= 1-(ao*10).clamp(0,1)*p['ambient_occlusion'][:,None,None,None]
        flash = torch.exp(-((self.x-p['flash_position'][:,0,None,None,None]).square()
                            +(self.y-p['flash_position'][:,1,None,None,None]).square())/.25)
        result += flash*p['flash_strength'][:,None,None,None]
        return result.clamp(.15,1.65)

    def deform(self, base_field, p, generator):
        """Invert the relief projection with Newton steps, checking convergence.

        Projection is a plane homography composed with x=u+h(u,v)*view_x,
        y=v+h(u,v)*view_y. Noninjective/nonconverged regions are invalid,
        rather than claiming that a height field solves folded-over surfaces.
        """
        packed,normals = self.surface(p,generator)
        shading = self.illumination(packed[:,:1],normals,p)
        wanted = torch.stack([base_field[...,0],base_field[...,1]*self.aspect],-1)
        uv = wanted.clone()
        vx,vy = p['relief_view'][:,0,None,None],p['relief_view'][:,1,None,None]
        for _ in range(self.inverse_iterations):
            field = torch.stack([uv[...,0],uv[...,1]/self.aspect],-1)
            values = F.grid_sample(packed,field,padding_mode='border',align_corners=False)
            z,hx,hy = values[:,0],values[:,1],values[:,2]
            fx,fy = uv[...,0]+z*vx-wanted[...,0],uv[...,1]+z*vy-wanted[...,1]
            determinant = 1+vx*hx+vy*hy
            safe = determinant.abs().clamp_min(.20)*torch.where(determinant<0,-1.,1.)
            dx = ((1+vy*hy)*fx-vx*hy*fy)/safe
            dy = ((1+vx*hx)*fy-vy*hx*fx)/safe
            uv -= torch.stack([dx,dy],-1).clamp(-.20,.20)
        field = torch.stack([uv[...,0],uv[...,1]/self.aspect],-1)
        values = F.grid_sample(packed,field,padding_mode='border',align_corners=False)
        z,hx,hy = values[:,0],values[:,1],values[:,2]
        residual = ((uv[...,0]+z*vx-wanted[...,0]).square()+(uv[...,1]+z*vy-wanted[...,1]).square()).sqrt()
        stable = ((1+vx*hx+vy*hy)>.15)&(residual<2/self.width)
        valid = stable[:,None].float()
        normal_image = F.grid_sample(normals,field,padding_mode='border',align_corners=False)
        normal_image = F.normalize(normal_image,dim=1)
        light_image = F.grid_sample(shading,field,padding_mode='border',align_corners=False)
        height_image = values[:,:1]
        asset_id = torch.full((len(p['asset_index']),),-1,device=self.device,dtype=torch.long)
        if len(self.asset_mask):
            ids = p['asset_index'].long()
            asset_field = F.grid_sample(self.asset_inverse_field[ids].permute(0,3,1,2),base_field,padding_mode='border',align_corners=False).permute(0,2,3,1)
            asset_mask = F.grid_sample(self.asset_mask[ids],base_field,align_corners=False)
            asset_normals = F.normalize(F.grid_sample(self.asset_normals[ids],base_field,align_corners=False),dim=1)
            asset_height = F.grid_sample(self.asset_height[ids],base_field,align_corners=False)
            # Bank normals are camera-space and retain their captured geometry.
            asset_light = p['ambient_light'][:,None,None,None].expand_as(light_image).clone()
            for index in range(3):
                direction = F.normalize(p['light_direction'][:,index],dim=-1)
                diffuse = (asset_normals*direction[:,:,None,None]).sum(1,keepdim=True).clamp_min(0)
                asset_light += diffuse*p['light_energy'][:,index,None,None,None]*p['light_color'][:,index,:,None,None]
            asset_light *= F.grid_sample(self.asset_shading[ids],base_field,align_corners=False)
            active = p['asset_active'][:,None,None,None].bool()
            field = torch.where(active.permute(0,2,3,1),asset_field,field)
            valid = torch.where(active,asset_mask,valid)
            normal_image = torch.where(active,asset_normals,normal_image)
            light_image = torch.where(active,asset_light,light_image)
            height_image = torch.where(active,asset_height,height_image)
            asset_id = torch.where(p['asset_active'].bool(),ids,asset_id)
        return {'field':field,'solver_valid':valid,'normals':normal_image,
                'height':height_image,'illumination':light_image,'asset_id':asset_id,
                'projection_residual':residual[:,None]}

    def background(self,p,generator):
        b = len(p['background_type'])
        x,y = self.x.expand(b,-1,-1,-1),self.y.expand(b,-1,-1,-1)
        angle = p['background_angle'][:,None,None,None]
        u,v = x*angle.cos()+y*angle.sin(),-x*angle.sin()+y*angle.cos()
        scale = p['background_scale'][:,None,None,None]
        low = F.interpolate(torch.rand(b,1,7,9,device=self.device,generator=generator),size=x.shape[-2:],mode='bicubic',align_corners=False)-.5
        fine = torch.randn(x.shape,device=self.device,generator=generator)*.008
        grain = torch.sin((v+low*.08)*scale*180)*.040+torch.sin(v*scale*53)*.024
        knot = torch.exp(-((u-.4).square()*8+(v+.18).square()*65))
        wood = grain-knot*.09*torch.sin((u.square()+v.square()).sqrt()*190)
        marble = -.12*torch.exp(-((torch.sin((u+low*.8+v*.17)*13)*8).square()))+low*.09
        fabric = .025*(torch.sin(u*230)+torch.sin(v*230))+low*.045
        metal = .022*torch.sin(v*260)+low*.08
        clinical = -.075*((torch.sin(x*12).abs()<.045)|(torch.sin(y*12).abs()<.045))+low*.025
        leather = low*.065+torch.sin(u*75)*torch.sin(v*79)*.017
        variations = [wood,marble,grain*.22+low*.018,fabric,metal,clinical,low*.009,leather]
        texture = torch.zeros_like(low)
        kind = p['background_type'][:,None,None,None]
        for index,variation in enumerate(variations):
            texture += variation*(kind==index)
        color = self.table_colors[p['background_type'].long()][:,:,None,None]
        image = (color+texture+fine)*p['background_tint'][:,:,None,None]
        return F.interpolate(image.clamp(0,1),size=(self.height,self.width),mode='bicubic',align_corners=False).clamp(0,1)

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
        b,_,h,w = paper_mask.shape
        size = self.x.shape[-2:]
        mask = F.interpolate(paper_mask,size=size,mode='area')
        base = torch.stack([self.x.expand(b,-1,-1,-1)[:,0],self.y.expand(b,-1,-1,-1)[:,0]/self.aspect],-1)
        result = torch.zeros_like(mask)
        relief = scene['height'].amax((-1,-2,-3))+.010
        for index in range(3):
            light = F.normalize(p['light_direction'][:,index],dim=-1)
            offset = light[:,:2]/light[:,2:].clamp_min(.2)*relief[:,None]
            field = base+torch.stack([offset[:,0],offset[:,1]/self.aspect],-1)[:,None,None]
            shifted = F.grid_sample(mask,field,align_corners=False)
            soft = F.avg_pool2d(shifted,5,1,2)
            softer = F.avg_pool2d(shifted,11,1,5)
            blend = (p['cast_shadow_softness']/.04).clamp(0,1)[:,None,None,None]
            result = torch.maximum(result,torch.lerp(soft,softer,blend))
        contact = F.avg_pool2d(F.max_pool2d(mask,3,1,1),3,1,1)*.65
        result = torch.maximum(result,contact)*p['cast_shadow_strength'][:,None,None,None]
        return F.interpolate(result,size=(h,w),mode='bilinear',align_corners=False)*(1-paper_mask)

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
