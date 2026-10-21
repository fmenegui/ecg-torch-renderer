from pathlib import Path
import importlib.metadata,json,os,shutil,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
RESULTS.mkdir(exist_ok=True)
def phase(name):
 (ROOT/'phase.json').write_text(json.dumps({'phase':name,'at':time.time()}))
 print('PHASE',name,flush=True)
def command(args,timeout=600,env=None):
 subprocess.run(args,cwd=ROOT,check=True,timeout=timeout,env=env)
phase('verify-gpu')
import torch
info={'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
assert torch.cuda.is_available(),'Real CUDA GPU required'
info.update(name=torch.cuda.get_device_name(),memory_bytes=torch.cuda.get_device_properties(0).total_memory)
x=torch.randn(512,512,device='cuda');y=x@x.T;torch.cuda.synchronize()
info['matmul_finite']=bool(torch.isfinite(y).all())
(RESULTS/'gpu.json').write_text(json.dumps(info,indent=2))
del x,y;torch.cuda.empty_cache()
phase('install-dependencies')
command([sys.executable,'-m','pip','install','--disable-pip-version-check','-e','.[digitization]','-r','experiment-requirements.txt'],timeout=900)
phase('unit-tests')
with (RESULTS/'unit-tests.log').open('w') as log:
 p=subprocess.run([sys.executable,'-m','unittest','tests.test_field_rectifier','tests.test_grid_point_quality','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 if p.returncode:raise RuntimeError('Unit checks failed')
phase('prepare-fresh-patients')
cohort=RESULTS/'cohort'
shutil.copytree(ROOT/'cohort',cohort)
shutil.copy2(ROOT/'excluded-patients.json',RESULTS/'excluded-patients.json')
from ecg_gpu.grid_benchmark import load_sources
records,fs=load_sources(cohort/'signals.npz')
excluded=json.loads((ROOT/'excluded-patients.json').read_text())
assert not {float(r['group_id'].split('patient-')[1]) for r in records}.intersection(excluded)
phase('verify-mitsuba-cuda')
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
scene=mi.load_dict({'type':'scene','integrator':{'type':'path'},'sensor':{'type':'perspective','film':{'type':'hdrfilm','width':16,'height':16}},'light':{'type':'constant','radiance':1.}})
image=mi.render(scene,spp=2)
assert bool(torch.isfinite(torch.tensor(image.numpy())).all())
phase('consolidated-detector-study')
command([sys.executable,'-u','examples/consolidated_grid_study.py','--signals',str(cohort/'signals.npz'),'--out',str(RESULTS),'--steps','1800'],timeout=8500)
phase('prepare-public-npj-source')
source=ROOT/'upstream-npj';env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
command(['git','init',str(source)],env=env)
command(['git','-C',str(source),'remote','add','origin','https://github.com/Ahus-AIM/Open-ECG-Digitizer.git'],env=env)
command(['git','-C',str(source),'fetch','--depth','1','origin','97a15087d4abcda843da8c58ee74b1d8f47e6f9a'],env=env)
command(['git','-C',str(source),'checkout','--detach','FETCH_HEAD'],env=env)
if not shutil.which('git-lfs'):
 command(['apt-get','update']);command(['apt-get','install','-y','git-lfs'])
command(['git','-C',str(source),'lfs','install','--local']);command(['git','-C',str(source),'lfs','pull'])
phase('consolidated-signal-study')
command([sys.executable,'-u','examples/consolidated_signal_study.py','--root',str(RESULTS),'--source',str(source)],timeout=1800)
versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','scipy','mitsuba','timm','pandas']}
(RESULTS/'packages.json').write_text(json.dumps(versions,indent=2))
phase('additional-frozen-reference-comparisons')
from ecg_gpu.digitization.artifacts import fetch_assets
fetch_assets(ROOT/'reference-assets')
command([sys.executable,'-u','examples/consolidated_references.py','--root',str(RESULTS),'--source',str(source),'--assets',str(ROOT/'reference-assets'),'--external',str(ROOT/'external/external.json')],timeout=1800)
command([sys.executable,'examples/summarize_consolidated_study.py','--root',str(RESULTS)],timeout=180)
phase('completed')
