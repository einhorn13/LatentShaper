
@echo off
title Latent Shaper
echo Starting Latent Shaper...
echo.

:: Check for virtual environment (optional but recommended)
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

:: Run the GUI
python gui_launcher.py

pause