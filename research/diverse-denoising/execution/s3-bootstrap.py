import base64,hashlib,hmac,http.server,json,os,signal,subprocess,sys,threading,time,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path('/workspace/ecg-diverse-v3')
ROOT.mkdir(parents=True,exist_ok=True)
SECRET=os.environ['ECG_EXPERIMENT_SECRET']
EXPECTED=os.environ['ECG_BUNDLE_SHA256']
STATE={'state':'ready','created_at':time.time()}
LOCK=threading.Lock()
try:
 GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True,timeout=20).strip()
except Exception as exc:GPU='unavailable: '+type(exc).__name__

def file_digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
 return h.hexdigest()

def monitor():
 STATE.update(state='running',started_at=time.time())
 code=-1
 try:
  with (ROOT/'job.log').open('w') as log:
   p=subprocess.Popen([sys.executable,'-u',str(ROOT/'job.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=24000)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    code=124
 except Exception as exc:STATE['error']=type(exc).__name__
 temporary=ROOT/'results.zip.part'
 with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for folder in ['results','inputs']:
   if (ROOT/folder).exists():
    for path in sorted((ROOT/folder).rglob('*')):
     if path.is_file() and not any(part in ('bank','test-predictions','reference-results','reference-inputs','wfdb') for part in path.relative_to(ROOT).parts) and not path.name.endswith('-recovery.pt') and path.name!='signals.npz':
      z.write(path,str(path.relative_to(ROOT)))
  for name in ['job.log','phase.json','experiment-requirements.txt']:
   if (ROOT/name).exists():z.write(ROOT/name,name)
 temporary.replace(ROOT/'results.zip')
 STATE.update(state='completed' if code==0 else 'failed',returncode=code,finished_at=time.time(),results_sha256=file_digest(ROOT/'results.zip'))

class Handler(http.server.BaseHTTPRequestHandler):
 def authorized(self):
  return hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+SECRET)
 def reply(self,status,body,content_type='application/json'):
  self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_GET(self):
  if not self.authorized():self.reply(401,b'{}');return
  if self.path=='/health':
   data=dict(STATE,gpu=GPU)
   if (ROOT/'results.zip').exists():data['results_bytes']=(ROOT/'results.zip').stat().st_size
   if (ROOT/'phase.json').exists():
    try:data['progress']=json.loads((ROOT/'phase.json').read_text())
    except Exception:pass
   
   for prefix in ('smoke', 'results'):
    progress=ROOT/prefix/'progress.json'
    if progress.exists():
     try:data[prefix+'_progress']=json.loads(progress.read_text())
     except Exception:pass
   download=ROOT/'download-progress.json'
   if download.exists():
    try:data['download_progress']=json.loads(download.read_text())
    except Exception:pass
   progress=ROOT/'results/progress.json'
   if progress.exists():
    try:data['study_progress']=json.loads(progress.read_text())
    except Exception:pass
   self.reply(200,json.dumps(data).encode());return
  if self.path=='/tail':
   path=ROOT/'job.log'
   data=''.join(path.read_text(errors='replace').splitlines(keepends=True)[-50:]) if path.exists() else ''
   self.reply(200,data.encode(),'text/plain');return
  if self.path.startswith('/export/'):
   name=self.path.split('/')[-1]
   path=ROOT/'exports'/name
   if not name.startswith('ecg-offline-bank-') or not name.endswith('.zip') or not path.is_file():self.reply(404,b'{}');return
   self.send_response(200);self.send_header('Content-Type','application/zip');self.send_header('Content-Length',str(path.stat().st_size));self.end_headers()
   with path.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):self.wfile.write(chunk)
   return
  if self.path=='/results' and (ROOT/'results.zip').exists():
   data=(ROOT/'results.zip').read_bytes();self.reply(200,data,'application/zip');return
  self.reply(404,b'{}')
 def do_POST(self):
  if not self.authorized():self.reply(401,b'{}');return
  if self.path.startswith('/chunk/'):
   with LOCK:
    if STATE['state']!='ready':self.reply(409,b'{}');return
    try:index=int(self.path.split('/')[-1])
    except ValueError:self.reply(400,b'{}');return
    size=int(self.headers.get('Content-Length','0'))
    if not 0<=index<60 or not 0<size<=30_000_000:self.reply(413,b'{}');return
    data=self.rfile.read(size)
    if len(data)!=size or hashlib.sha256(data).hexdigest()!=self.headers.get('X-Chunk-SHA256'):
     self.reply(400,b'{}');return
    folder=ROOT/'chunks';folder.mkdir(exist_ok=True)
    path=folder/f'{index:03d}'
    if path.exists() and path.read_bytes()!=data:self.reply(409,b'{}');return
    path.write_bytes(data)
    self.reply(200,json.dumps({'index':index,'bytes':size}).encode());return
  if self.path=='/commit':
   with LOCK:
    if STATE['state']!='ready':self.reply(409,b'{}');return
    count=int(self.headers.get('X-Chunk-Count','0'))
    if not 1<=count<=60:self.reply(400,b'{}');return
    assembled=ROOT/'assembled.zip'
    with assembled.open('wb') as stream:
     for index in range(count):
      part=ROOT/'chunks'/f'{index:03d}'
      if not part.exists():self.reply(400,b'{}');return
      stream.write(part.read_bytes())
    self.path='/upload'
    self.headers.replace_header('Content-Length',str(assembled.stat().st_size))
    self.rfile=assembled.open('rb')
  if self.path!='/upload':self.reply(404,b'{}');return
  with LOCK:
   if STATE['state']!='ready':self.reply(409,b'{"error":"already submitted"}');return
   size=int(self.headers.get('Content-Length','0'))
   if not 0<size<=1_800_000_000:self.reply(413,b'{}');return
   data=self.rfile.read(size)
   if len(data)!=size or hashlib.sha256(data).hexdigest()!=EXPECTED:
    self.reply(400,b'{"error":"bundle checksum mismatch"}');return
   archive=ROOT/'bundle.zip'
   archive.write_bytes(data)
   with zipfile.ZipFile(archive) as z:
    if sum(item.file_size for item in z.infolist())>4_000_000_000:
     self.reply(413,b'{}');return
    for item in z.infolist():
     path=PurePosixPath(item.filename)
     if path.is_absolute() or '..' in path.parts or ((item.external_attr>>16)&0o170000)==0o120000:
      self.reply(400,b'{"error":"invalid archive path"}');return
    z.extractall(ROOT)
    for item in z.infolist():
     if not item.is_dir():os.chmod(ROOT/item.filename,0o755 if ((item.external_attr>>16)&0o111) else 0o644)
   STATE['state']='submitted'
   threading.Thread(target=monitor,daemon=True).start()
   self.reply(202,b'{"submitted":true}')

http.server.ThreadingHTTPServer(('0.0.0.0',8000),Handler).serve_forever()
