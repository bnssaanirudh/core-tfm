#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-amd}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [[ "${INSTALL_ROCM_TORCH:-0}" == "1" ]]; then
  : "${ROCM_INDEX_URL:?Set ROCM_INDEX_URL to the URL produced by the official PyTorch selector, e.g. https://download.pytorch.org/whl/rocm6.4}"
  python -m pip install torch torchvision torchaudio --index-url "$ROCM_INDEX_URL"
fi

python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(
        "PyTorch is not installed. Install the ROCm build first, or rerun with "
        "INSTALL_ROCM_TORCH=1 and ROCM_INDEX_URL set. Original error: " + repr(exc)
    )
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.version.hip is None:
    raise SystemExit("This is not a ROCm/HIP PyTorch build. Refusing to continue on an AMD runner.")
if not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch is installed but no GPU is visible to torch.cuda. Check ROCm/driver permissions.")
print("GPU:", torch.cuda.get_device_name(0))
PY

# Install the project first without replacing the already-selected accelerator build.
python -m pip install -e ".[test]"
python -m pip install -r amd_gpu_runner/requirements.txt

# A dependency resolver may theoretically replace torch; verify again after all installs.
python - <<'PY'
import torch
print("Post-install Torch:", torch.__version__)
print("Post-install HIP:", torch.version.hip)
print("Post-install GPU available:", torch.cuda.is_available())
if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit(
        "The environment no longer has a working ROCm PyTorch build after dependency installation. "
        "Reinstall the ROCm torch wheel from the official PyTorch selector and rerun this script."
    )
PY

python -m ipykernel install --user --name core-tfm-amd --display-name "CoRe-TFM AMD ROCm" >/dev/null 2>&1 || true

echo
echo "AMD environment ready. Activate with: source $VENV_DIR/bin/activate"
echo "Next: export TABPFN_TOKEN=... && python amd_gpu_runner/preflight.py"
