# Launch the Autism Assistant Streamlit UI from the project root.
# Handles: venv creation, dependency installation, and launch.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$VenvDir = ".venv"
$VenvPython = Join-Path $VenvDir "Scripts" "python.exe"
$TorchIndex = "https://download.pytorch.org/whl/cpu"
$TorchVersion = "2.15.0"

# ---- Verify project root ----
if (-not (Test-Path "ui.py")) {
    Write-Host "[launch] ERROR: ui.py not found in $(Get-Location)" -ForegroundColor Red
    exit 1
}

# ---- Create venv if missing ----
if (-not (Test-Path $VenvPython)) {
    Write-Host "[setup] Creating virtual environment..."
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}

# ---- Install dependencies if streamlit is not importable ----
$streamlitOk = & $VenvPython -c "import streamlit" 2>$null
if (-not $streamlitOk -or $LASTEXITCODE -ne 0) {
    Write-Host "[setup] Installing dependencies (first run or missing packages)..."

    Write-Host "[setup] Installing CPU-only PyTorch $TorchVersion..."
    & $VenvPython -m pip install --index-url $TorchIndex "torch==$TorchVersion"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] ERROR: Failed to install PyTorch from the CPU index." -ForegroundColor Red
        exit 1
    }

    Write-Host "[setup] Installing project dependencies..."
    & $VenvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] ERROR: Failed to install project dependencies." -ForegroundColor Red
        exit 1
    }

    Write-Host "[setup] Verifying dependency graph..."
    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] ERROR: pip check failed — dependency graph is inconsistent." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[setup] Dependencies already installed — skipping installation."
}

# ---- Launch ----
Write-Host "[launch] Starting Autism Assistant UI..."
Write-Host "[launch] Model cache: see config.py CACHE_DIR (default: D:\autism_assistant_cache or .cache)"
& $VenvPython -m streamlit run ui.py
