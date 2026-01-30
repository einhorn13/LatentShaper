@echo off
setlocal enabledelayedexpansion

echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.10 or 3.11.
    pause
    exit /b
)

echo [INFO] Installing/Updating 'uv' package manager...
pip install uv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install uv.
    pause
    exit /b
)

echo.
echo [INFO] Creating virtual environment (.venv)...
uv venv .venv

echo.
echo [INFO] Installing PyTorch with CUDA 12.1 support...
REM Installing Torch specifically for CUDA to ensure GPU acceleration works
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [INFO] Installing dependencies from requirements.txt...
uv pip install -r requirements.txt

echo.
echo [SUCCESS] Installation complete!
echo You can now run the application using run.bat
pause