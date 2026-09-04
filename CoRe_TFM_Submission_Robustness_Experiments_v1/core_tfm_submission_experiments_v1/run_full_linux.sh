#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-$(pwd)}"
PY="${PYTHON:-python}"

if [[ -z "${TABPFN_TOKEN:-}" ]]; then
  echo "TABPFN_TOKEN is not set." >&2
  exit 2
fi

"$PY" "$(dirname "$0")/scripts/preflight_submission.py" --repo "$REPO"
"$PY" "$(dirname "$0")/scripts/plan_matrix.py" --preset full --output "$REPO/results/core_tfm_submission_full_v1/PLANNED_MATRIX.json"
"$PY" "$(dirname "$0")/scripts/run_suite.py" --repo "$REPO" --preset full --stage all
"$PY" "$(dirname "$0")/scripts/finalize_submission.py" --repo "$REPO" --preset full
