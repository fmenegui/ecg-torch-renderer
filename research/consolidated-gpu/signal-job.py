from pathlib import Path
import importlib.metadata,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
RESULTS.mkdir(exist_ok=True)
def phase(name):
 (ROOT/'phase.json').write_text(json.dumps({'phase':name,'at':time.time()}))
 print('PHASE',name,flush=True)
def command(args,timeout=600):
 subprocess.run(args,cwd=ROOT,check=True,timeout=timeout)
phase('verify-inference-gpu')
import torch
assert torch.cuda.is_available()
x=torch.randn(256,256,device='cuda');y=x@x.T;torch.cuda.synchronize()
assert torch.isfinite(y).all()
(RESULTS/'inference-gpu.json').write_text(json.dumps({'torch':str(torch.__version__),'cuda':torch.version.cuda,'name':torch.cuda.get_device_name(),'matmul_finite':True},indent=2))
del x,y;torch.cuda.empty_cache()
phase('install-offline-dependencies')
command([sys.executable,'-m','pip','install','--no-index','--find-links','wheels','--no-build-isolation','-e','.[digitization]','-r','experiment-requirements.txt'],timeout=300)
phase('verify-pinned-source-and-frozen-checkpoints')
source=ROOT/'upstream-npj'
os.environ['PATH']=str(ROOT/'bin')+os.pathsep+os.environ['PATH']
command(['git','-C',str(source),'lfs','install','--local'])
status=subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
if status:raise RuntimeError('Pinned source has tracked changes: '+status)
from ecg_gpu.digitization.artifacts import sha256_file,verify_assets
from examples.digitization_npj_probe import WEIGHTS,UPSTREAM_COMMIT
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==UPSTREAM_COMMIT
for filename,digest in WEIGHTS.items():assert sha256_file(source/'weights'/filename)==digest
for trial in json.loads((RESULTS/'trials.json').read_text()):assert sha256_file(RESULTS/trial['checkpoint'])==trial['checkpoint_sha256']
verify_assets(ROOT/'reference-assets')
phase('frozen-signal-extraction')
command([sys.executable,'-u','examples/resume_consolidated_signals.py','--root',str(RESULTS),'--source',str(source)],timeout=1800)
phase('matched-reference-and-external-transfer')
command([sys.executable,'-u','examples/consolidated_references.py','--root',str(RESULTS),'--source',str(source),'--assets',str(ROOT/'reference-assets'),'--external',str(ROOT/'external/external.json')],timeout=1800)
command([sys.executable,'examples/summarize_consolidated_study.py','--root',str(RESULTS)],timeout=180)
versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','scipy','mitsuba','timm','pandas']}
(RESULTS/'packages.json').write_text(json.dumps(versions,indent=2))
phase('completed')
