"""Benchmark the public image layer, profile visible tensors, and test a train step.

Timings use the real device. CPU instrumentation is not a CUDA measurement.
The allocation ledger counts public ATen results and excludes backend workspaces.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import weakref

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from .demo import demo_signals
from .gpu_module import ECGImageLayer
from .gpu_renderer import RenderConfig


PRESETS = {
    'fast': dict(height=256,width=384,antialias_samples=1,surface_resolution=32,inverse_iterations=3,shadow_samples=0),
    'balanced': dict(height=384,width=512,antialias_samples=2,surface_resolution=64,inverse_iterations=3,shadow_samples=2),
    'full': dict(height=768,width=1024,antialias_samples=4,surface_resolution=96,inverse_iterations=5,shadow_samples=8),
}


class TensorLedger(TorchDispatchMode):
    """Track unique result storages and aliases without inspecting tensor values.

    Visible live bytes exclude cuDNN/BLAS/compiler workspaces and allocator
    fragmentation. Created bytes are an allocation-volume proxy, not DRAM traffic.
    Instrumentation runs separately from latency measurements.
    """
    def __init__(self, constants):
        super().__init__()
        self.excluded={value.untyped_storage()._cdata for value in constants}
        self.references={}
        self.storages={}
        self.calls=Counter()
        self.bytes_by_op=Counter()
        self.live_bytes=0
        self.peak_bytes=0
        self.created_bytes=0

    def _drop(self, tensor_id, storage_id):
        self.references.pop(tensor_id,None)
        state=self.storages.get(storage_id)
        if state is None:
            return
        state[1]-=1
        if state[1]==0:
            self.live_bytes-=state[0]
            del self.storages[storage_id]

    def _track(self,value,operation):
        tensor_id=id(value)
        if tensor_id in self.references and self.references[tensor_id]() is value:
            return
        storage=value.untyped_storage()
        storage_id=storage._cdata
        if storage_id in self.excluded:
            return
        if storage_id not in self.storages:
            size=storage.nbytes()
            self.storages[storage_id]=[size,0]
            self.live_bytes+=size
            self.created_bytes+=size
            self.bytes_by_op[operation]+=size
            self.peak_bytes=max(self.peak_bytes,self.live_bytes)
        self.storages[storage_id][1]+=1
        self.references[tensor_id]=weakref.ref(value,lambda _:self._drop(tensor_id,storage_id))

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        operation=str(func)
        self.calls[operation]+=1
        output=func(*args,**(kwargs or {}))
        for value in tree_flatten(output)[0]:
            if isinstance(value,torch.Tensor):
                self._track(value,operation)
        return output

    def report(self):
        return {'visible_tensor_peak_bytes':self.peak_bytes,'visible_tensor_created_bytes':self.created_bytes,
                'aten_calls':sum(self.calls.values()),'top_calls':self.calls.most_common(12),
                'top_created_bytes':self.bytes_by_op.most_common(12),
                'scope':'public ATen result storages, unique aliases, excludes constant/input buffers and backend workspaces',
                'created_bytes_are_measured_dram_traffic':False,
                'cpu_ledger_is_cuda_memory_measurement':False}


def profile_regions(layer,signals):
    renderer=layer.renderer
    regions=[(renderer,'_curves'),(renderer,'_grid'),(renderer,'_text'),
             (renderer,'_stamp'),(renderer.document,'deform'),(renderer.document,'background'),
             (renderer.document,'cast_shadow'),(renderer.document,'object_shadow'),
             (renderer.camera,'optical_blur'),(renderer.camera,'jpeg')]
    saved=[]
    for owner,name in regions:
        original=getattr(owner,name)
        saved.append((owner,name,original))
        def wrapper(*args,_original=original,_name=name,**kwargs):
            with torch.profiler.record_function('ecg/'+_name):
                return _original(*args,**kwargs)
        setattr(owner,name,wrapper)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            image=layer(signals)
        del image
    finally:
        for owner,name,original in saved:
            setattr(owner,name,original)
    events=[event for event in profile.key_averages() if event.key.startswith('ecg/')]
    return [{'region':event.key,'cpu_total_ms':event.cpu_time_total/1000,'cpu_self_ms':event.self_cpu_time_total/1000,
             'calls':event.count} for event in sorted(events,key=lambda event:event.cpu_time_total,reverse=True)]


def measure(preset,batch_size,device,steps,warmup,ledger=True,regions=False,compile_raster=False,compile_scene=False):
    configuration=RenderConfig(profile='mixed',severity=.6,**PRESETS[preset])
    layer=ECGImageLayer(fs=500,config=configuration,seed=1973,compile_raster=compile_raster,compile_scene=compile_scene).to(device)
    values,_,_=demo_signals(fs=500,duration=10)
    signals=torch.tensor(values,dtype=torch.float32,device=device)[None].expand(batch_size,-1,-1).contiguous()
    synchronize=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    started=time.perf_counter()
    image=layer(signals)
    synchronize()
    cold_start_ms=(time.perf_counter()-started)*1000
    del image
    for _ in range(warmup):
        image=layer(signals)
        synchronize()
        del image
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    times=[]
    cuda_times=[]
    for _ in range(steps):
        synchronize()
        start_event=torch.cuda.Event(enable_timing=True) if device.type=='cuda' else None
        end_event=torch.cuda.Event(enable_timing=True) if device.type=='cuda' else None
        started=time.perf_counter()
        if start_event:
            start_event.record()
        image=layer(signals)
        if end_event:
            end_event.record()
        synchronize()
        times.append(time.perf_counter()-started)
        if start_event:
            cuda_times.append(start_event.elapsed_time(end_event))
        del image
    result={'preset':preset,'device':str(device),'batch_size':batch_size,'configuration':asdict(configuration),
            'compile_raster':compile_raster,'compile_scene':compile_scene,'first_call_ms':cold_start_ms,'first_call_includes_compilation_or_cache_loading':compile_raster or compile_scene,
            'steps':steps,'warmup':warmup,'latency_ms_median':float(np.median(times)*1000),
            'latency_ms_p90':float(np.quantile(times,.9)*1000),
            'images_per_second':batch_size*steps/sum(times),
            'output_bytes':batch_size*3*configuration.height*configuration.width*4,
            'input_bytes':signals.numel()*signals.element_size(),
            'cuda_event_ms_median':float(np.median(cuda_times)) if cuda_times else None,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
    if ledger and (compile_raster or compile_scene):
        result['ledger']={'unavailable':True,'reason':'ATen interception does not establish internal fused-kernel allocations. Use CUDA allocator peak on target hardware'}
    elif ledger:
        mode=TensorLedger([signals,*layer.buffers()])
        with mode:
            image=layer(signals)
        result['ledger']=mode.report()
        del image
    if regions and device.type=='cpu':
        result['cpu_profile_regions']=profile_regions(layer,signals)
    return result


def train_step(device,compile_raster=False,compile_scene=False):
    """Functional on-the-fly/AMP verification with a small artificial CNN."""
    torch.manual_seed(90)
    layer=ECGImageLayer(fs=500,config=RenderConfig(height=128,width=192,surface_resolution=32,shadow_samples=0),compile_raster=compile_raster,compile_scene=compile_scene).to(device)
    cnn=torch.nn.Sequential(torch.nn.Conv2d(3,8,3,stride=4),torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(8,2)).to(device)
    optimizer=torch.optim.AdamW(cnn.parameters(),lr=1e-3)
    pipeline=torch.nn.Sequential(layer,cnn)
    signals=torch.zeros(2,12,5000,device=device)
    classes=torch.tensor([0,1],device=device)
    before=[value.detach().clone() for value in cnn.parameters()]
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type,dtype=torch.float16 if device.type=='cuda' else torch.bfloat16):
        output=pipeline(signals)
        loss=torch.nn.functional.cross_entropy(output,classes)
    loss.backward()
    gradients=all(value.grad is not None and torch.isfinite(value.grad).all() for value in cnn.parameters())
    optimizer.step()
    changed=any(not torch.equal(a,b) for a,b in zip(before,cnn.parameters()))
    if not gradients or not changed:
        raise AssertionError('Image network did not complete the on-the-fly AMP train step')
    return {'device':str(device),'compile_raster':compile_raster,'compile_scene':compile_scene,'input_shape':list(signals.shape),'network_output_shape':list(output.shape),
            'amp_dtype':str(output.dtype),'finite_gradients':bool(gradients),'optimizer_changed_weights':changed,
            'signal_generation':'entirely artificial','cnn':'minimal functional example, not a clinical benchmark'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--presets',nargs='+',choices=list(PRESETS),default=list(PRESETS))
    parser.add_argument('--batches',nargs='+',type=int,default=[1,4])
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--ledger',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--regions',action='store_true')
    parser.add_argument('--compile-raster',action='store_true',help='Fuse deterministic raster blocks with torch.compile, first call may compile')
    parser.add_argument('--compile-scene',action='store_true',help='Fuse deterministic paper geometry, lighting and Newton updates')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. CPU results cannot establish GPU throughput')
    if min(args.batches+[args.steps,args.threads])<1 or args.warmup<0:
        parser.error('Positive batch sizes, steps and threads required')
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    args.out.mkdir(parents=True,exist_ok=True)
    report={'schema':'ecg-renderer-performance-audit-v1','device':str(device),'cuda_measured':device.type=='cuda',
            'torch_version':str(torch.__version__),'host':platform.machine(),'threads':args.threads,
            'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'timed_scope':'public ECGImageLayer.forward, signals already on device, includes per-call RNG management',
            'excluded_from_timing':['signal disk reading','signal transfer','font initialization','image encoding','CNN'],
            'profile_and_ledger_outside_timing':True,'compile_raster':args.compile_raster,'real_patient_records':0,'cases':[]}
    for preset in args.presets:
        for batch_size in args.batches:
            result=measure(preset,batch_size,device,args.steps,args.warmup,args.ledger,args.regions and batch_size==1,args.compile_raster,args.compile_scene)
            report['cases'].append(result)
            (args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({key:result[key] for key in ['preset','batch_size','latency_ms_median','images_per_second']}),flush=True)
    report['compile_scene']=args.compile_scene
    report['on_the_fly_amp_train_step']=train_step(device,args.compile_raster,args.compile_scene)
    root=Path(__file__).resolve().parents[1]
    report['source_sha256']={name:hashlib.sha256((root/'ecg_gpu'/name).read_bytes()).hexdigest()
                             for name in ['gpu_module.py','gpu_renderer.py','gpu_raster.py','gpu_document.py','gpu_appearance.py']}
    (args.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
