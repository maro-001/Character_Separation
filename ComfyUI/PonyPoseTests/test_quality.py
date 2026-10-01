import json,time
from pathlib import Path
import requests
import run_pony as r
from graph_utils import ui_workflow

while True:
 q=r.get('/queue')
 if not q['queue_running'] and not q['queue_pending']:break
 time.sleep(2)
requests.post(r.BASE+'/free',json={'unload_models':True,'free_memory':True},timeout=30).raise_for_status()
schedule=[('thibaud',.75,.75),('thibaud',1.,.7),('xinsir',.5,.75),('xinsir',1.,.7)]
definitions=r.get('/object_info')
results=json.loads((r.TEST/'results.json').read_text())
for cn,strength,end in schedule:
 name=f'pony_quality_{cn}_s{strength:g}_end{end:g}'
 if any(x['name']==name and x['status']=='success' for x in results):continue
 spec=dict(name=name,controlnet=cn,strength=strength,end_percent=end,seed=20261001,pose='source')
 print('QUALITY TEST',name,flush=True)
 p=r.workflow(spec)
 r.write(r.TEST/'workflows'/(name+'.api.json'),p)
 r.write(r.TEST/'workflows'/(name+'.json'),ui_workflow(p,definitions))
 h,seconds=r.execute(p);r.write(r.TEST/'history'/(name+'.json'),h)
 im=h['outputs']['7']['images'][0]
 row=dict(spec,status='success',seconds=seconds,image=str(r.ROOT/'ComfyUI/output'/im['subfolder']/im['filename']))
 results=[x for x in results if x['name']!=name]+[row];r.write(r.TEST/'results.json',results)
 print('OK',seconds,'seconds',flush=True)
r.evaluate();r.report()