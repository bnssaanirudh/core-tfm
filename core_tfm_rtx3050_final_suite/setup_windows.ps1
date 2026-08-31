param(
    [string]$RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $RepoPath).Path
$Venv = Join-Path $RepoPath ".venv"

Write-Host "Repository: $RepoPath"
Write-Host "Virtualenv: $Venv"

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & py "-$PythonVersion" -m venv $Venv
} else {
    & python -m venv $Venv
}

$Python = Join-Path $Venv "Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel

# CUDA-enabled PyTorch for Windows/NVIDIA. RTX 3050 is supported by CUDA builds.
& $Python -m pip install "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" --index-url https://download.pytorch.org/whl/cu128

Push-Location $RepoPath
try {
    & $Python -m pip install -e ".[test]"
    & $Python -m pip install -r (Join-Path $PSScriptRoot "requirements-extra.txt")
    & $Python -m pip check
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Environment installed."
Write-Host "In the SAME PowerShell window set your TabPFN token before running:"
Write-Host '$env:TABPFN_TOKEN = "YOUR_TOKEN_HERE"'
Write-Host "Then run:"
Write-Host ".\preflight_windows.ps1"
