import hashlib,json,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
TEST=Path(__file__).resolve().parent
meta=json.loads((TEST/'civitai_model.json').read_text(encoding='utf-8'))
v=next(x for x in meta['modelVersions'] if x['id']==290640)
f=next(x for x in v['files'] if x.get('primary'))
path=ROOT/'models/checkpoints'/f['name'];part=path.with_suffix('.safetensors.part')
size=round(f['sizeKB']*1024);digest=f['hashes']['SHA256'].lower()
if not path.exists() or path.stat().st_size!=size:
 for attempt in range(4):
  offset=part.stat().st_size if part.exists() else 0
  try:
   with requests.get(f['downloadUrl'],headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,90)) as response:
    response.raise_for_status()
    if offset and response.status_code!=206:offset=0
    done=offset;last=time.monotonic()
    with part.open('ab' if offset else 'wb') as out:
     for chunk in response.iter_content(8*1024*1024):
      if not chunk:continue
      out.write(chunk);done+=len(chunk)
      if time.monotonic()-last>15:print('Pony download',round(done/size*100,1),'%',flush=True);last=time.monotonic()
   assert part.stat().st_size==size,(part.stat().st_size,size)
   part.replace(path);break
  except (requests.RequestException,AssertionError) as e:
   print('Retry',attempt+1,str(e)[:200],flush=True)
   if attempt==3:raise
h=hashlib.sha256()
with path.open('rb') as src:
 while chunk:=src.read(16*1024*1024):h.update(chunk)
assert h.hexdigest()==digest
record=dict(model_id=257749,version_id=290640,file_id=f['id'],model_name=meta['name'],version=v['name'],filename=path.name,bytes=size,sha256=digest,source=f['downloadUrl'],installed_path=str(path))
(TEST/'download_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Pony SHA256 verified',path,flush=True)