"""Signal-to-image nn.Module, directly composable with image networks."""
import hashlib
import json
import math

import torch
from torch import nn

from .gpu_renderer import ECGRenderer, RenderConfig
from .gpu_parameters import Uniform, Choice, Bernoulli, ParameterSampler, parameter_tensor


class ECGImageLayer(nn.Module):
    """[B,12,T] mV -> [B,3,H,W] float32, no file I/O in forward.

    randomize=True changes the capture on every training call. Evaluation
    repeats the same capture seed. randomize=False repeats it in both modes.
    Fixed parameters are registered buffers and move once with .to(device).
    The renderer is a data transform with no gradient to the source signal.
    All parameters of the downstream image network remain trainable.
    """
    def __init__(self, fs=500., *, config=None, parameters=None, seed=1973,
                 randomize=True, renderer=None, photo=True, stream_id=0, **renderer_options):
        super().__init__()
        if not math.isfinite(float(fs)) or fs<=0:
            raise ValueError('Positive finite sampling rate required')
        if seed is not None and (not isinstance(seed,int) or seed<0):
            raise ValueError('seed must be a nonnegative integer or None')
        if type(photo) is not bool or type(stream_id) is not int or stream_id < 0:
            raise ValueError('photo must be bool and stream_id a nonnegative integer')
        if renderer is not None and (config is not None or renderer_options):
            raise ValueError('Pass renderer or config/renderer_options, not both')
        self.renderer=renderer or ECGRenderer(config or RenderConfig(),**renderer_options)
        self.fs=float(fs)
        self.register_buffer('_sampling_rate',torch.tensor(self.fs,device=self.renderer.device),persistent=False)
        self.seed=seed
        self.photo=photo
        self.stream_id=stream_id
        self._stream_offset=self._stream_seed_offset(stream_id)
        self.randomize=bool(randomize)
        self._calls=0
        self._rng_cache={}
        self._parameter_keys=[]
        self.samplers=nn.ModuleDict()
        prototypes=self.renderer.sample_parameters(1,generator=torch.Generator(device=self.renderer.device).manual_seed(0))
        for key,value in (parameters or {}).items():
            if key not in prototypes:
                raise ValueError(f'Unknown rendering parameter: {key}')
            if isinstance(value,(Uniform,Choice,Bernoulli)):
                self.samplers[key]=ParameterSampler(value,prototypes[key],lambda item: self.renderer.resolve_parameter(key,item))
                continue
            value=self.renderer.resolve_parameter(key,value)
            tensor=parameter_tensor(value,prototypes[key],name=key).clone()
            tensor.expand_as(prototypes[key])
            name='fixed_'+key
            self.register_buffer(name,tensor)
            self._parameter_keys.append(key)
        signature={'renderer':self.renderer.specification(),'fs':self.fs,'parameter_keys':sorted(self._parameter_keys),
                   'samplers':{key:value.specification() for key,value in self.samplers.items()}}
        self._signature=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()

    def get_extra_state(self):
        return {'schema':'ecg-image-layer-state-v1','signature':self._signature,
                'calls':self._calls,'seed':self.seed,'randomize':self.randomize,'compile_raster':self.renderer.compile_raster,
                'compile_scene':self.renderer.compile_scene,'photo':self.photo,'stream_id':self.stream_id}

    def set_extra_state(self,state):
        if state.get('schema')!='ecg-image-layer-state-v1' or state.get('signature')!=self._signature:
            raise ValueError('Image layer configuration/assets changed. Recreate the same layer before resuming')
        if state.get('compile_raster',False)!=self.renderer.compile_raster:
            raise ValueError('Rasterizer backend changed. Recreate the same compile_raster setting before resuming')
        if state.get('compile_scene',False)!=self.renderer.compile_scene:
            raise ValueError('Scene backend changed. Recreate the same compile_scene setting before resuming')
        if not isinstance(state['calls'],int) or state['calls']<0:
            raise ValueError('Invalid image-layer RNG state')
        self._calls=state['calls']
        self.seed=state['seed']
        self.randomize=state['randomize']
        if type(state.get('photo', True)) is not bool or type(state.get('stream_id', 0)) is not int or state.get('stream_id', 0) < 0:
            raise ValueError('Invalid capture or RNG stream state')
        self.photo=state.get('photo', True)
        self.stream_id=state.get('stream_id', 0)
        self._stream_offset=self._stream_seed_offset(self.stream_id)

    @staticmethod
    def _stream_seed_offset(stream_id):
        if stream_id == 0:
            return 0
        return int.from_bytes(hashlib.blake2b(str(stream_id).encode(), digest_size=8,
                              person=b'ecg-stream-v1').digest(), 'little') % (2**63-1)

    def _generator(self,generator):
        if generator is not None:
            return generator
        if self.seed is None and self.training and self.randomize:
            return None
        base=0 if self.seed is None else self.seed
        offset=self._calls if self.training and self.randomize else 0
        if self.training and self.randomize:
            self._calls+=1
        device=self.renderer.device
        if device not in self._rng_cache:
            self._rng_cache[device]=torch.Generator(device=device)
        return self._rng_cache[device].manual_seed((base+self._stream_offset+1000003*offset)%(2**63-1))

    def render_batch(self,signals,fs=None,*,lengths=None,parameters=None,generator=None,
                     stage='all',photo=None,**options):
        photo=self.photo if photo is None else photo
        fixed={key:getattr(self,'fixed_'+key) for key in self._parameter_keys}
        rng=self._generator(generator)
        for key,sampler in self.samplers.items():
            fixed[key]=sampler(signals.shape[0],rng)
        fixed.update(parameters or {})
        return self.renderer(signals,self._sampling_rate if fs is None else fs,lengths=lengths,
                             parameters=fixed,generator=rng,stage=stage,photo=photo,**options)

    def forward(self,signals,fs=None,*,lengths=None,parameters=None,generator=None):
        return self.render_batch(signals,fs,lengths=lengths,parameters=parameters,
                                 generator=generator,stage='image').image


