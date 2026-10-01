@echo off
set PYTHONUTF8=1
"%~dp0..\.venv\Scripts\python.exe" -B "%~dp0PonyPoseTests\start_work.py" --open
if errorlevel 1 pause
