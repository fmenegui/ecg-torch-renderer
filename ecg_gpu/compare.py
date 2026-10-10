"""Paired performance comparison against the frozen standalone v0.2.1 ZIP.

Synthetic signals, identical seed sequences and cyclic backend order. No
image encoding or CNN is timed. CUDA results require a CUDA device.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch

from .demo import demo_signals
from .audit import PRESETS
from .gpu_module import ECGImageLayer
from .gpu_renderer import RenderConfig


def load_legacy(archive,directory):
    destination=Path(directory)/'ecg_v021'
    destination.mkdir()
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise ValueError('Legacy archive checksum failed')
        for name in z.namelist():
            if '/ecg_gpu/' in name and name.endswith('.py'):
                filename=Path(name).name
                (destination/filename).write_bytes(z.read(name))
    spec=importlib.util.spec_from_file_location('_ecg_v021',destination/'__init__.py',submodule_search_locations=[str(destination)])
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    if module.__version__!='0.2.1':
        raise ValueError('Expected the frozen 0.2.1 standalone package')
    return module


def measure_case(legacy,preset,batch_size,device,steps,warmup):
    options=dict(profile='mixed',severity=.6,**PRESETS[preset])
    layers={
        'v021':legacy.ECGImageLayer(fs=500,config=legacy.RenderConfig(**options),seed=1973).to(device),
        'eager':ECGImageLayer(fs=500,config=RenderConfig(**options),seed=1973).to(device),
        'compiled':ECGImageLayer(fs=500,config=RenderConfig(**options),seed=1973,compile_raster=True).to(device),
    }
    values,_,_=demo_signals(fs=500,duration=10)
    signals=torch.tensor(values,dtype=torch.float32,device=device)[None].expand(batch_size,-1,-1).contiguous()
    synchronize=lambda:torch.cuda.synchronize(device) if device.type=='cuda' else None
    cold={}
    for name,layer in layers.items():
        started=time.perf_counter()
        image=layer(signals)
        synchronize()
        cold[name]=(time.perf_counter()-started)*1000
        del image
        for _ in range(warmup):
            image=layer(signals)
            synchronize()
            del image
        layer._calls=100
    # Pixel comparison is outside timing and uses equal public-layer call seeds.
    images={name:layer(signals) for name,layer in layers.items()}
    errors={name:float((value-images['v021']).abs().max()) for name,value in images.items() if name!='v021'}
    del images
    if errors['eager']>2e-6 or errors['compiled']>5e-5:
        raise AssertionError(f'Paired image fidelity changed: {errors}')
    for layer in layers.values():
        layer._calls=0
    latencies={name:[] for name in layers}
    peak={name:[] for name in layers}
    event_times={name:[] for name in layers}
    names=list(layers)
    for iteration in range(steps):
        order=names[iteration%len(names):]+names[:iteration%len(names)]
        for name in order:
            synchronize()
            baseline=torch.cuda.memory_allocated(device) if device.type=='cuda' else None
            if device.type=='cuda':
                torch.cuda.reset_peak_memory_stats(device)
            a=torch.cuda.Event(enable_timing=True) if device.type=='cuda' else None
            b=torch.cuda.Event(enable_timing=True) if device.type=='cuda' else None
            started=time.perf_counter()
            if a is not None:a.record()
            image=layers[name](signals)
            if b is not None:b.record()
            synchronize()
            latencies[name].append((time.perf_counter()-started)*1000)
            if a is not None:
                event_times[name].append(a.elapsed_time(b))
                peak[name].append(torch.cuda.max_memory_allocated(device)-baseline)
            del image
    backends={name:{'median_ms':float(np.median(values)),'p90_ms':float(np.quantile(values,.9)),
                   'images_s':batch_size*steps*1000/sum(values),'first_call_ms':cold[name],
                   'cuda_event_ms_median':float(np.median(event_times[name])) if event_times[name] else None,
                   'cuda_peak_additional_allocated_bytes':max(peak[name]) if peak[name] else None}
              for name,values in latencies.items()}
    return {'preset':preset,'width':options['width'],'height':options['height'],'batch_size':batch_size,
            'steps':steps,'warmup':warmup,'backends':backends,'max_rgb_error_vs_v021':errors,
            'eager_speedup_vs_v021':backends['v021']['median_ms']/backends['eager']['median_ms'],
            'compiled_speedup_vs_v021':backends['v021']['median_ms']/backends['compiled']['median_ms']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--presets',nargs='+',choices=list(PRESETS),default=list(PRESETS))
    p.add_argument('--batches',nargs='+',type=int,default=[1,4])
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--warmup',type=int,default=3)
    args=p.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    if min(args.batches+[args.steps])<1 or args.warmup<0:p.error('Invalid measurement counts')
    torch.set_num_threads(2)
    device=torch.device(args.device)
    args.out.mkdir(parents=True,exist_ok=True)
    result={'schema':'ecg-raster-paired-benchmark-v1','device':str(device),'cuda_measured':device.type=='cuda',
            'torch_version':str(torch.__version__),'threads':2,'legacy_zip_sha256':hashlib.sha256(args.legacy.read_bytes()).hexdigest(),
            'comparison':'cyclic backend order, identical public-layer seed sequences, encoding/CNN excluded',
            'compilation_excluded_from_warm_timings':True,'real_patient_records':0,'cases':[]}
    with tempfile.TemporaryDirectory() as directory:
        legacy=load_legacy(args.legacy,directory)
        for preset in args.presets:
            for batch_size in args.batches:
                case=measure_case(legacy,preset,batch_size,device,args.steps,args.warmup)
                result['cases'].append(case)
                (args.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(case),flush=True)
    root=Path(__file__).resolve().parents[1]
    result['source_sha256']={name:hashlib.sha256((root/'ecg_gpu'/name).read_bytes()).hexdigest()
                            for name in ['gpu_renderer.py','gpu_raster.py','gpu_module.py']}
    (args.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
