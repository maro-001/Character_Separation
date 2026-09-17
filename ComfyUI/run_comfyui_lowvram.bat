@echo off
REM ComfyUI launcher - low VRAM mode (for SDXL + segmentation models together)
cd /d "%~dp0..\ComfyUI"
set PYTHONUTF8=1
call ".venv\Scripts\python.exe" main.py --preview-method auto --lowvram --listen 127.0.0.1 --port 8188 %*
pause
