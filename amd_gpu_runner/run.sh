#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV_DIR="${VENV_DIR:-.venv-amd}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing $VENV_DIR. Run: bash amd_gpu_runner/setup_amd.sh" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

: "${TABPFN_TOKEN:?Export TABPFN_TOKEN before running this script.}"

python -m amd_gpu_runner.preflight
python -m amd_gpu_runner.run_queue "$@"
