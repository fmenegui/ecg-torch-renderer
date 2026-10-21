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
phase('verify-public-s3-connectivity')
import urllib.request
s3='https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/'
with urllib.request.urlopen(s3+'ptbxl_database.csv',timeout=30) as response:
 assert response.status==200
 assert hashlib.sha256(response.read()).hexdigest()=='7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b'
phase('offline-dependencies')
command([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-build-isolation','-e','.[digitization,pathtrace]','-r','experiment-requirements.txt'],timeout=900)
phase('focused-tests')
with (RESULTS/'focused-tests.log').open('w') as log:
 p=subprocess.run([sys.executable,'-m','unittest','tests.test_denoising_grid','tests.test_local_cells','tests.test_hybrid_rectifier','tests.test_prepare_grid_benchmark','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 if p.returncode:raise RuntimeError('Focused tests failed')
phase('download-public-ptbxl-s3-to-persistent-cache')
command([sys.executable,'-m','pip','install','--quiet','awscli'],timeout=300)
cache=Path('/workspace/datasets/ptb-xl/1.0.3');cache.mkdir(parents=True,exist_ok=True)
aws_config=ROOT/'aws-public-config'
aws_config.write_text('[default]\ns3 =\n    max_concurrent_requests = 32\n    max_queue_size = 2000\n')
env=dict(os.environ,AWS_CONFIG_FILE=str(aws_config),AWS_EC2_METADATA_DISABLED='true')
download=subprocess.Popen([sys.executable,'-m','awscli','s3','sync','s3://physionet-open/ptb-xl/1.0.3/',str(cache),
 '--no-sign-request','--exclude','*','--include','records100/*','--include','*.csv','--include','LICENSE.txt','--no-progress','--only-show-errors'],env=env)
while download.poll() is None:
 status={'files':sum(1 for p in cache.rglob('*') if p.is_file()),'expected_records':21799,'time':time.time()}
 (ROOT/'download-progress.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
 time.sleep(15)
assert download.returncode==0,'Public S3 acquisition failed'
phase('verify-public-metadata-and-public-exclusion-ids')
from ecg_gpu.digitization.artifacts import sha256_file
import csv,urllib.request
expected_metadata_sha256='7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b'
shutil.copy2(cache/'ptbxl_database.csv',ROOT/'ptbxl_database.csv')
assert sha256_file(ROOT/'ptbxl_database.csv')==expected_metadata_sha256
with (ROOT/'ptbxl_database.csv').open() as stream:public_rows=list(csv.DictReader(stream))
public_ids={str(int(float(row['patient_id']))) for row in public_rows}
excluded=json.loads((ROOT/'public-ptbxl-exclusion-ids.json').read_text())
assert len(public_rows)==21799 and set(excluded).issubset(public_ids)
(ROOT/'excluded-patients.json').write_text(json.dumps(excluded))
(RESULTS/'public-data-audit.json').write_text(json.dumps({'source':'https://physionet.org/content/ptb-xl/1.0.3/','license':'CC BY 4.0','metadata_sha256':expected_metadata_sha256,'public_exclusion_identifiers_verified':len(excluded),'private_images_used':False},indent=2))
phase('prepare-1280-public-ptbxl-patients')
from examples.prepare_grid_benchmark import prepare
flat=ROOT/'wfdb-flat';flat.mkdir(exist_ok=True)
for path in (cache/'records100').rglob('*'):
 if path.suffix in ('.hea','.dat'):
  dest=flat/path.name
  if not dest.exists():dest.symlink_to(path)
cohort=RESULTS/'cohort'
if not (cohort/'sources.json').exists():
 prepare(ROOT/'ptbxl_database.csv',cohort,train=1024,validation=128,test=128,seed=20261022,
        wfdb_cache=flat, excluded_patients=json.loads((ROOT/'excluded-patients.json').read_text()))
phase('small-end-to-end-smoke')
import numpy as np
with np.load(cohort/'signals.npz') as data:
 selected=[]
 for split in ('train','validation','test'):
  selected.extend(np.flatnonzero(data['splits']==split)[:2].tolist())
 smoke_cohort=ROOT/'smoke-cohort';smoke_cohort.mkdir(exist_ok=True)
 np.savez_compressed(smoke_cohort/'signals.npz',**{key:(data[key][selected] if key in ('signals','source_ids','group_ids','splits') else data[key]) for key in data.files})
smoke_source=json.loads((cohort/'sources.json').read_text())
smoke_source['signals_sha256']=sha256_file(smoke_cohort/'signals.npz')
(smoke_cohort/'sources.json').write_text(json.dumps(smoke_source))
smoke_code="import torch;from pathlib import Path;import examples.diverse_grid_study as s;torch.set_num_threads(4);s.SEEDS=(1973,);s.main(Path('smoke-cohort'),Path('smoke'),steps=20,batch_size=4)"
command([sys.executable,'-u','-c',smoke_code],timeout=1800)
smoke=ROOT/'smoke'
assert json.loads((smoke/'progress.json').read_text())['phase']=='completed'
(RESULTS/'smoke-evidence.json').write_text(json.dumps({'status':'passed','images':48,'steps_per_model':20,'trials':json.loads((smoke/'trials.json').read_text()),'note':'Engineering smoke only. These patients also appear in the full study and are not an independent benchmark.'},indent=2))
phase('gpu-render-train-evaluate')
command([sys.executable,'-u','examples/diverse_grid_study.py','--cohort',str(cohort),'--root',str(RESULTS),'--steps','10000','--batch-size','4','--resume'],timeout=21500)
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
