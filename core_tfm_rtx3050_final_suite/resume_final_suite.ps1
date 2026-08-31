param([string]$RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")))
# The runner checks COMPLETE.json/fold_results.csv and skips completed variants,
# so resume is intentionally the same deterministic pipeline.
& (Join-Path $PSScriptRoot "run_final_suite.ps1") -RepoPath $RepoPath
