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
echo [1] Run Video Analysis on sample_traffic.mp4 (Tracking + Cyber Bounding Boxes)
echo [2] Run Image Analysis on sample_road.jpg (CLI + Window Preview)
echo [3] Launch Interactive Web Dashboard (Streamlit)
echo [4] Run on Live Webcam (Camera 0)
echo [5] Exit
echo.
set /p choice="Enter option [1-5]: "

if "%choice%"=="1" (
    echo Running full video road analysis on sample_traffic.mp4...
    .\.venv\Scripts\python.exe main.py --source sample_traffic.mp4 --view
    pause
) else if "%choice%"=="2" (
    echo Running road analysis on sample_road.jpg...
    .\.venv\Scripts\python.exe main.py --source sample_road.jpg --view
    pause
) else if "%choice%"=="3" (
    echo Launching Streamlit Web App...
    .\.venv\Scripts\streamlit.exe run app.py
) else if "%choice%"=="4" (
    echo Launching Live Webcam Analysis...
    .\.venv\Scripts\python.exe main.py --source 0 --view
) else (
    echo Exiting.
)
