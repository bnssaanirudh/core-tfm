param(
    [string]$RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [ValidateSet("paper","multi_seed","context")][string]$Profile = "paper"
)
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $RepoPath).Path
$Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not $env:TABPFN_TOKEN) { throw 'TABPFN_TOKEN is not set in this PowerShell session.' }
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TOKENIZERS_PARALLELISM = "false"
Push-Location $PSScriptRoot
try {
    & $Python -m core_tfm_rtx3050_suite.run_matrix --repo $RepoPath --profile $Profile
} finally { Pop-Location }
