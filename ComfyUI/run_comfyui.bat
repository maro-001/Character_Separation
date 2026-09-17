@echo off
REM ComfyUI launcher - Character Separation Research
cd /d "%~dp0..\ComfyUI"
set PYTHONUTF8=1
call ".venv\Scripts\python.exe" main.py --preview-method auto --listen 127.0.0.1 --port 8188 %*
pause
