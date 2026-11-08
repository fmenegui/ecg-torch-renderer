from pathlib import Path
import json,subprocess,sys,time,os,shutil,zipfile,hashlib,importlib.metadata
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
def phase(name):
 (ROOT/'phase.json').write_text(json.dumps({'phase':name,'time':time.time()}));print('PHASE',name,flush=True)
def command(args,timeout=600):subprocess.run(args,cwd=ROOT,check=True,timeout=timeout)
phase('verify-real-cuda')
import torch
assert torch.cuda.is_available()
torch.set_num_threads(4)
x=torch.randn(256,256,device='cuda');y=x@x.T;torch.cuda.synchronize();assert torch.isfinite(y).all()
(RESULTS/'gpu.json').write_text(json.dumps({'device':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'matmul_finite':True},indent=2))
del x,y
phase('offline-dependencies')
command([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-build-isolation','-e','.[digitization,pathtrace]','-r','experiment-requirements.txt'],timeout=900)
phase('focused-tests')
with (RESULTS/'focused-tests.log').open('w') as log:
 p=subprocess.run([sys.executable,'-m','unittest','tests.test_denoising_grid','tests.test_local_cells','tests.test_hybrid_rectifier','tests.test_prepare_grid_benchmark','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 if p.returncode:raise RuntimeError('Focused tests failed')
phase('verify-public-metadata-and-public-exclusion-ids')
from ecg_gpu.digitization.artifacts import sha256_file
import csv,urllib.request
cached=Path('/workspace/ecg-experiment/ptbxl_database.csv')
expected_metadata_sha256='7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b'
if cached.is_file() and sha256_file(cached)==expected_metadata_sha256:
 shutil.copy2(cached,ROOT/'ptbxl_database.csv')
else:
 with urllib.request.urlopen('https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv',timeout=120) as response,(ROOT/'ptbxl_database.csv').open('wb') as stream:
  shutil.copyfileobj(response,stream)
assert sha256_file(ROOT/'ptbxl_database.csv')==expected_metadata_sha256
with (ROOT/'ptbxl_database.csv').open() as stream:public_rows=list(csv.DictReader(stream))
public_ids={str(int(float(row['patient_id']))) for row in public_rows}
excluded=json.loads((ROOT/'public-ptbxl-exclusion-ids.json').read_text())
assert len(public_rows)==21799 and set(excluded).issubset(public_ids)
(ROOT/'excluded-patients.json').write_text(json.dumps(excluded))
(RESULTS/'public-data-audit.json').write_text(json.dumps({'source':'https://physionet.org/content/ptb-xl/1.0.3/','license':'CC BY 4.0','metadata_sha256':expected_metadata_sha256,'public_exclusion_identifiers_verified':len(excluded),'private_images_used':False},indent=2))
phase('prepare-1280-public-ptbxl-patients')
from examples.prepare_grid_benchmark import prepare
cohort=RESULTS/'cohort'
prepare(ROOT/'ptbxl_database.csv',cohort,train=1024,validation=128,test=128,seed=20261022,
        excluded_patients=json.loads((ROOT/'excluded-patients.json').read_text()))
phase('gpu-render-train-evaluate')
command([sys.executable,'-u','examples/diverse_grid_study.py','--cohort',str(cohort),'--root',str(RESULTS),'--steps','10000','--batch-size','4'],timeout=21500)
phase('fetch-public-frozen-reference-assets')
try:
 command(['apt-get','update'],timeout=180)
 command(['apt-get','install','-y','git-lfs'],timeout=180)
 env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1')
 subprocess.run(['git','clone','https://github.com/Ahus-AIM/Open-ECG-Digitizer.git',str(ROOT/'upstream-npj')],env=env,check=True,timeout=180)
 command(['git','-C',str(ROOT/'upstream-npj'),'checkout','97a15087d4abcda843da8c58ee74b1d8f47e6f9a'])
 command(['git','-C',str(ROOT/'upstream-npj'),'lfs','install','--local'])
 command(['git','-C',str(ROOT/'upstream-npj'),'lfs','pull'],timeout=600)
 from ecg_gpu.digitization.artifacts import fetch_assets
 fetch_assets(ROOT/'reference-assets')
except Exception as exc:
 (RESULTS/'reference-acquisition-error.json').write_text(json.dumps({'status':'asset-acquisition-failed','error':str(exc),'algorithm_failure':False},indent=2))
phase('frozen-reference-image-checks')
import numpy as np
from PIL import Image
from ecg_gpu.digitization.artifacts import sha256_file
bank=json.loads((RESULTS/'bank-manifest.json').read_text())
rows=[r for r in bank if r['split']=='test'][:32]
inputs=RESULTS/'reference-inputs';inputs.mkdir(exist_ok=True);images=[]
for row in rows:
 with np.load(RESULTS/row['file']) as d:rgb=d['image'].copy()
 image=inputs/(row['key']+'.png');Image.fromarray(rgb).save(image)
 images.append({'id':row['key'],'file':image.name,'sha256':sha256_file(image)})
(inputs/'images.json').write_text(json.dumps(images,indent=2))
command([sys.executable,'-u','examples/rectification_references.py','--images',str(inputs/'images.json'),'--out',str(RESULTS/'reference-results'),'--source',str(ROOT/'upstream-npj'),'--assets',str(ROOT/'reference-assets'),'--device','cuda'],timeout=1800)
phase('qualitative-gallery')
import io
selected=json.loads((RESULTS/'selected.json').read_text())
predictions=RESULTS/'test-predictions'/selected['name']
gallery=RESULTS/'gallery';gallery.mkdir(exist_ok=True)
for row in rows:
 with np.load(RESULTS/row['file']) as data:
  for key in ('image','clean'):
   Image.fromarray(data[key]).save(gallery/(row['key']+'-'+key+'.jpg'),quality=92)
 for kind in ('restored','rectified'):
  source=predictions/(row['key']+'-'+kind+'.png')
  if source.exists():Image.open(source).convert('RGB').save(gallery/(row['key']+'-'+kind+'.jpg'),quality=92)
phase('package-offline-bank')
exports=ROOT/'exports';exports.mkdir(exist_ok=True)
parts=[];z=None;size=0;part=0
for row in bank:
 f=RESULTS/row['file']
 if z is None or size+f.stat().st_size>80_000_000:
  if z is not None:z.close()
  name=f'ecg-offline-bank-{part:03d}.zip';part+=1;size=0
  z=zipfile.ZipFile(exports/name,'w',compression=zipfile.ZIP_STORED)
  parts.append({'name':name,'files':0})
 z.write(f,row['file']);size+=f.stat().st_size;parts[-1]['files']+=1
if z is not None:z.close()
for row in parts:
 f=exports/row['name'];row.update(bytes=f.stat().st_size,sha256=sha256_file(f))
(RESULTS/'offline-bank-parts.json').write_text(json.dumps(parts,indent=2))
(RESULTS/'packages.json').write_text(json.dumps({name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','scipy','mitsuba','pandas','timm']},indent=2))
phase('completed')
