param([string]$RepoPath = (Resolve-Path "."))
$ErrorActionPreference = "Stop"
$RepoPath = (Resolve-Path $RepoPath).Path
$Python = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

if (-not $env:TABPFN_TOKEN) {
  throw "TABPFN_TOKEN is not set in this PowerShell session."
}

& $Python "$PSScriptRoot\scripts\preflight_submission.py" --repo $RepoPath
& $Python "$PSScriptRoot\scripts\plan_matrix.py" --preset full --output "$RepoPath\results\core_tfm_submission_full_v1\PLANNED_MATRIX.json"
& $Python "$PSScriptRoot\scripts\run_suite.py" --repo $RepoPath --preset full --stage all
& $Python "$PSScriptRoot\scripts\finalize_submission.py" --repo $RepoPath --preset full
