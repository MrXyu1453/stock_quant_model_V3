@echo off
title Stock Quant Platform - http://127.0.0.1:8050
chcp 65001 >nul
cd /d "%~dp0"

rem Prefer the project virtual environment; fall back to system Python if missing
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

rem If the app is already running, just open the browser
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri http://127.0.0.1:8050 -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    echo App is already running. Opening browser...
    start "" http://127.0.0.1:8050
    timeout /t 2 /nobreak >nul
    exit /b 0
)

rem Wait for the server to come up, then open the browser automatically
start "" /min powershell -NoProfile -Command "for($i=0;$i -lt 30;$i++){try{Invoke-WebRequest http://127.0.0.1:8050 -UseBasicParsing -TimeoutSec 1|Out-Null; Start-Process 'http://127.0.0.1:8050'; break}catch{Start-Sleep -Seconds 1}}"

"%PYTHON_EXE%" run_app.py

echo.
echo Server exited. Press any key to close...
pause >nul
