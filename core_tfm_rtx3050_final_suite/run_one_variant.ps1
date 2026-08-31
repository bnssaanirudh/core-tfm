param(
    [Parameter(Mandatory=$true)][int]$Seed,
    [Parameter(Mandatory=$true)][int]$TrainLimit,
    [ValidateSet("multi_seed","context","manual")][string]$Group = "manual",
    [string]$RepoPath = (Resolve-Path (Join-Path $PSScriptRoot ".."))
)
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $RepoPath).Path
$Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not $env:TABPFN_TOKEN) { throw 'TABPFN_TOKEN is not set.' }
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
Push-Location $PSScriptRoot
try {
    & $Python -m core_tfm_rtx3050_suite.run_variant --repo $RepoPath --group $Group --seed $Seed --train-limit $TrainLimit
} finally { Pop-Location }
