import argparse,json
import run_pony as r
from graph_utils import ui_workflow
parser=argparse.ArgumentParser();parser.add_argument('case');args=parser.parse_args()
results=json.loads((r.TEST/'results.json').read_text())
original=next(x for x in results if x['name']==args.case)
definitions=r.get('/object_info')
for pose,seed in [('source',20261002),('edited',20261001)]:
 name=args.case+('_confirm_seed20261002' if pose=='source' else '_edited_arm')
 if any(x['name']==name and x['status']=='success' for x in results):continue
 spec={k:original[k] for k in ['controlnet','strength','end_percent']}
 spec.update(name=name,pose=pose,seed=seed)
 print('CONFIRM',name,flush=True)
 p=r.workflow(spec);r.write(r.TEST/'workflows'/(name+'.api.json'),p);r.write(r.TEST/'workflows'/(name+'.json'),ui_workflow(p,definitions))
 h,seconds=r.execute(p);r.write(r.TEST/'history'/(name+'.json'),h)
 im=h['outputs']['7']['images'][0]
 row=dict(spec,status='success',seconds=seconds,image=str(r.ROOT/'ComfyUI/output'/im['subfolder']/im['filename']))
 results=[x for x in results if x['name']!=name]+[row];r.write(r.TEST/'results.json',results);print('OK',seconds,'seconds',flush=True)
r.evaluate();r.report(args.case)