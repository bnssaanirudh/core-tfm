param([string]$RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")))
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $RepoPath).Path
$Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
Push-Location $PSScriptRoot
try {
    & $Python -m core_tfm_rtx3050_suite.aggregate --repo $RepoPath
    & $Python -m core_tfm_rtx3050_suite.rare_class --repo $RepoPath
    & $Python -m core_tfm_rtx3050_suite.completion_gate --repo $RepoPath
    & $Python -m core_tfm_rtx3050_suite.make_paper_bundle --repo $RepoPath
} finally { Pop-Location }
