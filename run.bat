
@echo off
title Z-Image Turbo LoRA Studio
echo Starting Z-Image Turbo LoRA Studio...
echo.

:: Check for virtual environment (optional but recommended)
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

:: Run the GUI
python gui_launcher.py

pause