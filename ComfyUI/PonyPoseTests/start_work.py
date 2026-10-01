import argparse,json,os,subprocess,time,webbrowser
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
TEST=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8188'
parser=argparse.ArgumentParser();parser.add_argument('--open',action='store_true');args=parser.parse_args()
try:
 response=requests.get(BASE+'/system_stats',timeout=5);response.raise_for_status()
except requests.RequestException:
 env=os.environ.copy();env['PYTHONUTF8']='1';env['HF_HUB_DISABLE_TELEMETRY']='1'
 cmd=[str(ROOT/'.venv/Scripts/python.exe'),'-B','main.py','--preview-method','none','--listen','127.0.0.1','--port','8188','--disable-auto-launch','--lowvram']
 out=(TEST/'server.stdout.log').open('a',encoding='utf-8');err=(TEST/'server.stderr.log').open('a',encoding='utf-8')
 proc=subprocess.Popen(cmd,cwd=ROOT/'ComfyUI',env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
 (TEST/'server.json').write_text(json.dumps(dict(pid=proc.pid,command=cmd,url=BASE),indent=2),encoding='utf-8')
 for _ in range(120):
  try:
   response=requests.get(BASE+'/system_stats',timeout=3);response.raise_for_status();break
  except requests.RequestException:time.sleep(1)
 else:raise TimeoutError('Work server failed to start; check PonyPoseTests/server.stderr.log')
print('Work server ready',BASE,flush=True)
if args.open:webbrowser.open(BASE)