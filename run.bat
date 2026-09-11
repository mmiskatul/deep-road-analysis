@echo off
echo ===================================================
echo        DEEP ROAD ANALYSIS & SCENE PERCEPTION
echo ===================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment (.venv) not found!
    echo Please run: python -m venv .venv and install requirements.
    pause
    exit /b 1
)

echo Select Mode:
echo [1] Run Standard Video Analysis on sample_traffic.mp4
echo [2] Run LinkedIn Format Video (1:1 1080x1080 with Title Cards & Telemetry)
echo [3] Run Image Analysis on sample_road.jpg (CLI + Window Preview)
echo [4] Launch Interactive Web Dashboard (Streamlit)
echo [5] Run on Live Webcam (Camera 0)
echo [6] Exit
echo.
set /p choice="Enter option [1-6]: "

if "%choice%"=="1" (
    echo Running standard video road analysis...
    .\.venv\Scripts\python.exe main.py --source sample_traffic.mp4 --view
    pause
) else if "%choice%"=="2" (
    echo Generating LinkedIn showcase video (1080x1080)...
    .\.venv\Scripts\python.exe main.py --source sample_traffic.mp4 --linkedin --view
    pause
) else if "%choice%"=="3" (
    echo Running road analysis on sample_road.jpg...
    .\.venv\Scripts\python.exe main.py --source sample_road.jpg --view
    pause
) else if "%choice%"=="4" (
    echo Launching Streamlit Web App...
    .\.venv\Scripts\streamlit.exe run app.py
) else if "%choice%"=="5" (
    echo Launching Live Webcam Analysis...
    .\.venv\Scripts\python.exe main.py --source 0 --view
) else (
    echo Exiting.
)
