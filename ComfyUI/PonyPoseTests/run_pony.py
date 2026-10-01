import argparse,copy,csv,hashlib,html,json,shutil,statistics,time
from pathlib import Path
import requests
from graph_utils import node,ui_workflow
from pose_metrics import score_pose,points

ROOT=Path(__file__).resolve().parents[2]
TEST=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8188'
CHECKPOINT='ponyDiffusionV6XL_v6StartWithThisOne.safetensors'
PROMPT='score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, rating_safe, source_anime, two adult male knights, 2boys, full body, both characters fully visible, full plate armor, castle courtyard'
NEGATIVE=''
SEEDS=[20261001,20261002]

def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def get(path):
 r=requests.get(BASE+path,timeout=30);r.raise_for_status();return r.json()

def execute(p):
 begin=time.monotonic();resp=requests.post(BASE+'/prompt',json={'prompt':p},timeout=30);resp.raise_for_status();d=resp.json()
 if d.get('node_errors'):raise RuntimeError(d['node_errors'])
 pid=d['prompt_id']
 while time.monotonic()-begin<600:
  h=get('/history/'+pid)
  if pid in h:
   item=h[pid]
   if item['status']['status_str']!='success':raise RuntimeError(item['status'])
   return item,round(time.monotonic()-begin,2)
  time.sleep(1)
 raise TimeoutError(pid)

def specs():
 out=[]
 for seed in SEEDS:
  out.append(dict(name=f'pony_baseline_seed{seed}',controlnet='',strength=0.,seed=seed,pose='source'))
 for cn,strengths in [('thibaud',[1.,1.5]),('xinsir',[1.5,2.])]:
  for strength in strengths:
   for seed in SEEDS:out.append(dict(name=f'pony_{cn}_s{strength:g}_seed{seed}',controlnet=cn,strength=strength,seed=seed,pose='source'))
 for cn,strength in [('thibaud',1.5),('xinsir',2.)]:
  out.append(dict(name=f'pony_edited_{cn}_s{strength:g}',controlnet=cn,strength=strength,seed=SEEDS[0],pose='edited'))
 return out

def workflow(spec):
 raw=(TEST/('pose.json' if spec['pose']=='source' else 'edited_pose.json')).read_text(encoding='utf-8')
 pose=dict(show_body=True,show_face=True,show_hands=True,resolution_x=1024,pose_marker_size=12,face_marker_size=3,hand_marker_size=2,hands_scale=1.,body_scale=1.,head_scale=1.,overall_scale=1.,scalelist_behavior='poses',match_scalelist_method='loop extend',only_scale_pose_index=99,POSE_JSON=raw)
 p={'1':node('CheckpointLoaderSimple',dict(ckpt_name=CHECKPOINT),'Pony Diffusion V6 XL'),
 '10':node('CLIPSetLastLayer',dict(clip=['1',1],stop_at_clip_layer=-2),'CLIP Skip 2'),
 '2':node('CLIPTextEncode',dict(clip=['10',0],text=PROMPT),'Positive prompt'),
 '3':node('CLIPTextEncode',dict(clip=['10',0],text=NEGATIVE),'Negative prompt'),
 '4':node('EmptyLatentImage',dict(width=1024,height=1024,batch_size=1)),
 '40':node('OpenposeEditorNode',pose,'pose (1).json 본 편집기'),
 '41':node('PreviewImage',dict(images=['40',0]),'적용할 본 미리보기'),
 '6':node('VAEDecodeTiled',dict(samples=['5',0],vae=['1',2],tile_size=256,overlap=32,temporal_size=64,temporal_overlap=8)),
 '7':node('SaveImage',dict(images=['6',0],filename_prefix='PonyPoseTests/'+spec['name']))}
 positive,negative=['2',0],['3',0]
 if spec['controlnet']:
  p['8']=node('ControlNetLoader',dict(control_net_name=spec['controlnet']+'_openpose_sdxl.safetensors'))
  p['9']=node('ControlNetApplyAdvanced',dict(positive=positive,negative=negative,control_net=['8',0],image=['40',0],strength=spec['strength'],start_percent=0.,end_percent=spec.get('end_percent',1.),vae=['1',2]))
  positive,negative=['9',0],['9',1]
 p['5']=node('KSampler',dict(model=['1',0],positive=positive,negative=negative,latent_image=['4',0],seed=spec['seed'],steps=25,cfg=7.,sampler_name='euler_ancestral',scheduler='normal',denoise=1.))
 return p

def render_pose(edited=False):
 spec=dict(name='render',controlnet='',strength=0.,seed=SEEDS[0],pose='edited' if edited else 'source')
 p={'40':workflow(spec)['40'],'42':node('SaveImage',dict(images=['40',0],filename_prefix='PonyPoseTests/'+('edited_skeleton' if edited else 'source_skeleton')))}
 h,_=execute(p);write(TEST/('edited_pose_render.json' if edited else 'source_pose_render.json'),h)

def run():
 manifest=json.loads((TEST/'download_manifest.json').read_text(encoding='utf-8'))
 assert (ROOT/'models/checkpoints'/CHECKPOINT).stat().st_size==manifest['bytes']
 for port in [8189,8188]:
  q=requests.get(f'http://127.0.0.1:{port}/queue',timeout=10).json()
  if not q['queue_running'] and not q['queue_pending']:
   requests.post(f'http://127.0.0.1:{port}/free',json={'unload_models':True,'free_memory':True},timeout=30).raise_for_status()
 definitions=get('/object_info')
 assert CHECKPOINT in definitions['CheckpointLoaderSimple']['input']['required']['ckpt_name'][0]
 write(TEST/'test_plan.json',specs())
 write(TEST/'settings.json',dict(checkpoint=CHECKPOINT,prompt=PROMPT,negative=NEGATIVE,width=1024,height=1024,steps=25,cfg=7.,sampler='euler_ancestral',scheduler='normal',clip_skip=2,server=BASE,source='D:/Download/pose (1).json',sha256=hashlib.sha256((TEST/'pose_source_original.json').read_bytes()).hexdigest()))
 render_pose();render_pose(True)
 results=json.loads((TEST/'results.json').read_text(encoding='utf-8')) if (TEST/'results.json').exists() else []
 completed={r['name'] for r in results if r['status']=='success'}
 for i,spec in enumerate(specs(),1):
  if spec['name'] in completed:continue
  print('TEST',i,'/',len(specs()),spec['name'],flush=True)
  p=workflow(spec);write(TEST/'workflows'/(spec['name']+'.api.json'),p)
  ui=ui_workflow(p,definitions);write(TEST/'workflows'/(spec['name']+'.json'),ui)
  row=dict(spec)
  try:
   h,seconds=execute(p);write(TEST/'history'/(spec['name']+'.json'),h)
   im=h['outputs']['7']['images'][0]
   row.update(status='success',seconds=seconds,image=str(ROOT/'ComfyUI/output'/im['subfolder']/im['filename']))
   print('OK',seconds,'seconds',flush=True)
  except Exception as e:
   row.update(status='error',error=str(e));print('ERROR',str(e)[:500],flush=True)
  results=[x for x in results if x['name']!=row['name']]+[row];write(TEST/'results.json',results)
 print('GENERATED',sum(x['status']=='success' for x in results),'/',len(specs()),flush=True)

def evaluate():
 results=json.loads((TEST/'results.json').read_text(encoding='utf-8'))
 inputdir=ROOT/'ComfyUI/input/PonyPoseTests_eval';inputdir.mkdir(exist_ok=True)
 for i,row in enumerate(results,1):
  if row['status']!='success' or row.get('metrics'):continue
  print('EVALUATE',i,'/',len(results),row['name'],flush=True)
  path=Path(row['image']);shutil.copy2(path,inputdir/path.name)
  p={'1':node('LoadImage',dict(image='PonyPoseTests_eval/'+path.name)),
     '2':node('DWPreprocessor',dict(image=['1',0],detect_hand='disable',detect_body='enable',detect_face='disable',resolution=1024,bbox_detector='yolox_l.onnx',pose_estimator='dw-ll_ucoco_384.onnx',scale_stick_for_xinsr_cn='disable')),
     '3':node('PreviewImage',dict(images=['2',0]))}
  h,_=execute(p);estimate=json.loads(h['outputs']['2']['openpose_json'][0]);write(TEST/'detected_poses'/(row['name']+'.json'),estimate)
  reference=json.loads((TEST/('pose.json' if row['pose']=='source' else 'edited_pose.json')).read_text(encoding='utf-8'))
  row['metrics']=score_pose(reference,estimate)
  if row['pose']=='edited':
   index=row['metrics']['assignment'][0]
   detected=points(estimate[0]['people'][index],1024,1024) if index<len(estimate[0]['people']) else []
   target=points(reference[0]['people'][0],1024,1024)
   row['edited_joint_error_pixels']={str(j):round(((detected[j][0]-target[j][0])**2+(detected[j][1]-target[j][1])**2)**.5*1024,2) if j<len(detected) and detected[j][2]>0 else None for j in [3,4]}
  write(TEST/'results.json',results);print(row['metrics'],flush=True)

def report(recommended_case=None):
 results=json.loads((TEST/'results.json').read_text(encoding='utf-8'));groups={}
 for row in results:
  if not row.get('metrics') or row['pose']!='source':continue
  groups.setdefault((row['controlnet'],row['strength'],row.get('end_percent',1.)),[]).append(row)
 summary=[]
 for (cn,strength,end),rows in groups.items():
  body=statistics.mean(x['metrics']['pck_body'] for x in rows);arms=statistics.mean(x['metrics']['pck_arms'] for x in rows)
  summary.append(dict(controlnet=cn,strength=strength,end_percent=end,tests=len(rows),body=round(body,2),arms=round(arms,2),combined=round(.7*arms+.3*body,2)))
 summary.sort(key=lambda x:x['combined'],reverse=True);write(TEST/'summary.json',summary)
 if recommended_case is None:
  best=next(x for x in summary if x['controlnet'])
  candidates=[x for x in results if x['pose']=='source' and x['controlnet']==best['controlnet'] and x['strength']==best['strength'] and x.get('end_percent',1.)==best['end_percent']]
  recommended_case=max(candidates,key=lambda x:.7*x['metrics']['pck_arms']+.3*x['metrics']['pck_body'])['name']
 recommended=next(x for x in results if x['name']==recommended_case)
 write(TEST/'recommendation.json',recommended)
 for suffix in ['.json','.api.json']:shutil.copy2(TEST/'workflows'/(recommended_case+suffix),TEST/('Pony_Pose_Work'+suffix))
 user=ROOT/'ComfyUI/user/default/workflows';user.mkdir(exist_ok=True,parents=True)
 shutil.copy2(TEST/'Pony_Pose_Work.json',user/'Pony_Pose_Work.json')
 edited=next((x for x in results if x['pose']=='edited' and x['controlnet']==recommended['controlnet'] and x['strength']==recommended['strength'] and x.get('end_percent',1.)==recommended.get('end_percent',1.)),None)
 if edited:
  shutil.copy2(TEST/'workflows'/(edited['name']+'.json'),TEST/'Pony_Edited_Arm_Demo.json');shutil.copy2(TEST/'Pony_Edited_Arm_Demo.json',user/'Pony_Edited_Arm_Demo.json')
 with (TEST/'results.csv').open('w',encoding='utf-8-sig',newline='') as out:
  fields=['name','controlnet','strength','end_percent','seed','pose','status','seconds','pck_body','pck_arms','people_detected','image']
  writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
  for row in results:writer.writerow({k:row.get(k,row.get('metrics',{}).get(k)) for k in fields})
 def link(path):return '../../'+Path(path).relative_to(ROOT).as_posix()
 cards=[]
 for row in results:
  if row['status']!='success':continue
  m=row.get('metrics',{})
  cards.append(f'<article><h3>{html.escape(row["name"])}</h3><p>{row["pose"]} / seed {row["seed"]} / {row["seconds"]}s / 몸 {m.get("pck_body","?")}% / 팔 {m.get("pck_arms","?")}%</p><a href="{link(row["image"])}"><img loading="lazy" src="{link(row["image"])}"></a><a href="workflows/{row["name"]}.json">워크플로</a></article>')
 table=''.join(f'<tr><td>{x["controlnet"] or "제어 없음"}</td><td>{x["strength"]}</td><td>{x["end_percent"]}</td><td>{x["tests"]}</td><td>{x["body"]}%</td><td>{x["arms"]}%</td></tr>' for x in summary)
 skeletons=[]
 for title,name in [('원본 본','source_pose_render.json'),('팔을 올린 수정 본','edited_pose_render.json')]:
  h=json.loads((TEST/name).read_text(encoding='utf-8'));im=h['outputs']['42']['images'][0];path=ROOT/'ComfyUI/output'/im['subfolder']/im['filename'];skeletons.append(f'<article><h3>{title}</h3><img src="{link(path)}"></article>')
 page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Work Pony 본 테스트</title><style>body{font-family:system-ui,sans-serif;background:#f5f6f8;color:#17212b;max-width:1450px;margin:auto;padding:28px}p{line-height:1.7}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}article{background:white;padding:14px;border-radius:12px;margin:12px 0}h3{font-size:16px;overflow-wrap:anywhere}img{width:100%;border-radius:8px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#1460ac}@media(max-width:850px){.grid{grid-template-columns:1fr 1fr}}@media(max-width:520px){.grid{grid-template-columns:1fr}}</style><h1>원본 Work — Pony Diffusion V6 XL 본 테스트</h1>'''
 page+=f'<p>모델: V6 (start with this one), CLIP Skip 2, Euler a 25 steps / CFG 7 / normal / 1024×1024. 두 사람의 원본 pose (1).json과 팔꿈치·손목을 올린 수정 본을 사용했습니다. <a href="https://civitai.com/models/257749/pony-diffusion-v6-xl">모델 출처</a></p><p>추천 워크플로: {recommended["controlnet"]} 강도 {recommended["strength"]}, end_percent {recommended.get("end_percent",1.)}, seed {recommended["seed"]}. <a href="Pony_Pose_Work.json">워크플로</a> · <a href="results.csv">전체 CSV</a> · <a href="README.md">사용법</a></p><h2>기준 본</h2><div class="grid">'+''.join(skeletons)+'</div><h2>원본 본 — 두 시드 평균</h2><p>DWPose로 검출한 18개 몸 관절의 PCK@0.05입니다. 이미지 대각선의 5%(약 72px) 이내를 일치로 계산합니다. 팔 점수는 어깨·팔꿈치·손목만 비교합니다. 얼굴·손가락은 점수에 포함하지 않으며, 갑옷 때문에 검출이 실패할 수 있어 이미지도 확인해야 합니다.</p><table><tr><th>ControlNet</th><th>강도</th><th>횟수</th><th>몸 관절</th><th>팔 관절</th></tr>'+table+'</table><h2>생성 결과</h2><div class="grid">'+''.join(cards)+'</div></html>'
 (TEST/'comparison.html').write_text(page,encoding='utf-8')
 readme=f'''# Work — Pony 본 테스트

[모델 출처](https://civitai.com/models/257749/pony-diffusion-v6-xl)의 **V6 (start with this one)**을 `Work/models/checkpoints/{CHECKPOINT}`에 설치하고 제작자 SHA256과 대조했습니다. LoRA는 사용하지 않았습니다. 포즈 모델은 Work의 `models/controlnet`에 설치한 Thibaud/Xinsir OpenPose SDXL입니다. 기존 Character_Separation 워크플로는 유지했습니다.

## 실행과 본 편집

1. http://127.0.0.1:8188 을 열고 Ctrl+F5로 새로고침합니다. 서버가 꺼져 있으면 Work의 `run_comfyui.bat`를 실행합니다.
2. [Pony_Pose_Work.json](Pony_Pose_Work.json)을 화면에 드래그하거나 워크플로 목록에서 `Pony_Pose_Work`를 엽니다.
3. 본 편집기 노드 40을 우클릭하고 `Open in Openpose Editor`를 선택합니다.
4. 관절을 수정한 뒤 `ControlNet에 자세 보내기` / `Send pose to ControlNet`를 누르고 Run/실행합니다. 본 미리보기는 실행 후 갱신됩니다.
5. 워크플로를 저장하면 수정 본도 함께 저장됩니다. `Reload reference pose`는 해당 워크플로의 기준 본으로 되돌립니다.

기본 파일에는 `D:/Download/pose (1).json` 원본을 넣었습니다. CLIP Skip 2, Euler a 25 steps, CFG 7, normal, 1024×1024, 고정 시드입니다. 추천 ControlNet: **{recommended['controlnet']}, strength {recommended['strength']}, end_percent {recommended.get('end_percent',1.)}**.

[전체 비교 이미지](comparison.html) · [측정 결과](results.csv) · [상세 JSON](results.json) · [모델 다운로드·해시 기록](download_manifest.json). 제어 없음, Thibaud 강도 1/1.5, Xinsir 강도 1.5/2를 원본 본과 두 시드로 비교하고, Thibaud 1.5와 Xinsir 2로 팔을 올린 수정 본을 비교했습니다.

PCK 점수는 몸 관절이 허용 거리 안에 들어온 비율이며 손가락·얼굴 정확도를 보장하지 않습니다. 갑옷·헬멧의 검출 오류도 있을 수 있습니다. 생성 이미지: `Work/ComfyUI/output/PonyPoseTests`. 각 테스트의 UI/API 워크플로는 `workflows` 폴더에 있습니다.
'''
 (TEST/'README.md').write_text(readme,encoding='utf-8')
 print('SUMMARY',json.dumps(summary),flush=True);print('RECOMMENDED',recommended_case,flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['run','evaluate','report','all']);parser.add_argument('--recommended-case');args=parser.parse_args()
 if args.stage in ['run','all']:run()
 if args.stage in ['evaluate','all']:evaluate()
 if args.stage in ['report','all']:report(args.recommended_case)