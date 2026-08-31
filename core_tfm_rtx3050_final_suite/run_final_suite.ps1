param([string]$RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")))
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $RepoPath).Path
$Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Missing .venv. Run setup_windows.ps1 first." }
if (-not $env:TABPFN_TOKEN) { throw 'TABPFN_TOKEN is not set in this PowerShell session.' }
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TOKENIZERS_PARALLELISM = "false"

Push-Location $PSScriptRoot
try {
    Write-Host "=== 1/5 PRE-FLIGHT ==="
    & $Python -m core_tfm_rtx3050_suite.preflight --repo $RepoPath --strict-token

    Write-Host "=== 2/5 CPU / CONTROLLED EVIDENCE ==="
    & $Python -m core_tfm_rtx3050_suite.run_cpu_experiments --repo $RepoPath

    Write-Host "=== 3/5 REAL-TFM GPU MATRIX ==="
    & $Python -m core_tfm_rtx3050_suite.run_matrix --repo $RepoPath --profile paper

    Write-Host "=== 4/5 AGGREGATION + RARE CLASS ==="
    & $Python -m core_tfm_rtx3050_suite.aggregate --repo $RepoPath
    & $Python -m core_tfm_rtx3050_suite.rare_class --repo $RepoPath

    Write-Host "=== 5/5 COMPLETION GATE + PAPER BUNDLE ==="
    & $Python -m core_tfm_rtx3050_suite.completion_gate --repo $RepoPath
    & $Python -m core_tfm_rtx3050_suite.make_paper_bundle --repo $RepoPath
} finally {
    Pop-Location
}
