@echo off
rem Starts Quantlab on this PC (with the local AI) and opens it in the browser.
rem From your other devices on Tailscale: https://pp01.tail48bb31.ts.net
cd /d "%~dp0.."
curl -s -m 2 http://127.0.0.1:8501/_stcore/health >nul 2>&1
if errorlevel 1 (
    start "Quantlab" /min .venv\Scripts\python.exe -m streamlit run app/streamlit_app.py ^
        --server.headless true --server.address 127.0.0.1 --server.port 8501
    timeout /t 8 /nobreak >nul
)
start "" http://localhost:8501
