# Launch the Autism Assistant Streamlit UI from the project root.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

Write-Host "Starting Autism Assistant UI..."
Write-Host "Open the local URL shown by Streamlit (usually http://localhost:8501)"
streamlit run ui.py
