@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_VERSION=2.15.0"

rem ---- Resolve project root (already done by cd /d "%~dp0") ----
if not exist "ui.py" (
    echo [launch] ERROR: ui.py not found in %CD%
    echo [launch] Ensure this script is in the project root.
    pause
    exit /b 1
)

rem ---- Create venv if missing ----
if not exist "%VENV_PYTHON%" (
    echo [setup] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [setup] ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

rem ---- Install dependencies if streamlit is not importable ----
"%VENV_PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing dependencies (first run or missing packages)...

    echo [setup] Installing CPU-only PyTorch %TORCH_VERSION%...
    "%VENV_PYTHON%" -m pip install --index-url %TORCH_INDEX% "torch==%TORCH_VERSION%"
    if errorlevel 1 (
        echo [setup] ERROR: Failed to install PyTorch from the CPU index.
        pause
        exit /b 1
    )

    echo [setup] Installing project dependencies...
    "%VENV_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [setup] ERROR: Failed to install project dependencies.
        pause
        exit /b 1
    )

    echo [setup] Verifying dependency graph...
    "%VENV_PYTHON%" -m pip check
    if errorlevel 1 (
        echo [setup] ERROR: pip check failed — dependency graph is inconsistent.
        pause
        exit /b 1
    )
) else (
    echo [setup] Dependencies already installed — skipping installation.
)

rem ---- Launch ----
echo [launch] Starting Autism Assistant UI...
echo [launch] Model cache: see config.py CACHE_DIR (default: D:\autism_assistant_cache or .cache)
"%VENV_PYTHON%" -m streamlit run ui.py

pause
