param([string]$RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")))
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $RepoPath).Path
$Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Missing .venv. Run setup_windows.ps1 first." }
if (-not $env:TABPFN_TOKEN) { throw 'TABPFN_TOKEN is not set. In this PowerShell window run: $env:TABPFN_TOKEN="..."' }
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
Push-Location $PSScriptRoot
try {
    & $Python -m core_tfm_rtx3050_suite.preflight --repo $RepoPath --strict-token
} finally { Pop-Location }
