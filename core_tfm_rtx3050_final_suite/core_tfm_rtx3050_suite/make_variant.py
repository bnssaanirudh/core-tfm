from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

from core_tfm.robustness_runner import load_notebook, patch_q1_notebook

from .common import git_head, resolve_paths, set_low_vram_environment, variant_id


def _local_setup_source(*, root: Path, output_root: Path, run_id: str, seed: int, train_limit: int, test_limit: int) -> str:
    protocol_revision = f"rtx3050_cuda_seed{seed}_train{train_limit}_test{test_limit}"
    return f'''# Local NVIDIA/CUDA setup replacing the Colab-only Cell 2.
FULL_Q1_RUN = True
RUN_ID = {run_id!r}
PROTOCOL_REVISION = {protocol_revision!r}

from pathlib import Path
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

ROOT = Path({str(root)!r}).resolve()
DRIVE_BASE = Path({str(output_root)!r}).resolve()
RUN = DRIVE_BASE / RUN_ID
MODEL_CACHE = DRIVE_BASE / "model_cache"
RUN.mkdir(parents=True, exist_ok=True)
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODEL_CACHE / "huggingface")
os.environ["TABPFN_MODEL_CACHE_DIR"] = str(MODEL_CACHE / "tabpfn")
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")

FREEZE_PATH = RUN / "frozen_source_revisions.json"

def run(command, *, cwd=None):
    print("RUN:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)

head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
if FREEZE_PATH.exists():
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("core_tfm_commit") != head:
        raise RuntimeError(
            f"Run is frozen to {{frozen.get('core_tfm_commit')}} but repository HEAD is {{head}}. "
            "Checkout the frozen commit or use a new run directory."
        )
else:
    frozen = {{"core_tfm_commit": head}}

frozen.update({{
    "protocol_revision": PROTOCOL_REVISION,
    "execution_platform": "windows_local_nvidia_cuda",
    "bounded_context_protocol": {{"max_train_rows": {train_limit}, "max_test_rows": {test_limit}}},
    "hardware_target": "RTX 3050 Laptop 4GB",
}})
FREEZE_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not visible to PyTorch. Run the suite preflight first.")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3))
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

tabpfn_token = os.environ.get("TABPFN_TOKEN", "").strip()
if not tabpfn_token:
    raise RuntimeError(
        "TABPFN_TOKEN is not set in this PowerShell session. Set $env:TABPFN_TOKEN before running."
    )
os.environ["TABPFN_TOKEN"] = tabpfn_token

# Headless license/token validation when the installed TabPFN exposes this API.
try:
    from tabpfn.browser_auth import ensure_license_accepted, verify_token
    from tabpfn.errors import TabPFNLicenseError
    from tabpfn.settings import settings
    token_status = verify_token(tabpfn_token, settings.tabpfn.auth_api_url)
    if token_status is not True:
        raise RuntimeError("TABPFN_TOKEN is invalid or the Prior Labs server is unreachable.")
    try:
        ensure_license_accepted(hf_repo_id="tabpfn_3")
    except TabPFNLicenseError as exc:
        if "browser login is disabled" in str(exc):
            raise RuntimeError("Accept the TabPFN-3 license in the Prior Labs account and rerun.") from exc
        raise
except ImportError:
    pass

ENV_PATH = RUN / "environment_metadata.json"
env_payload = {{
    "source_commit": head,
    "platform": platform.platform(),
    "python": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    "seed": {seed},
    "train_limit": {train_limit},
    "test_limit": {test_limit},
}}
ENV_PATH.write_text(json.dumps(env_payload, indent=2), encoding="utf-8")
print("Persistent run directory:", RUN)
print("Protocol revision:", PROTOCOL_REVISION)
'''


def build_variant(
    *,
    root: Path,
    output_root: Path,
    run_id: str,
    seed: int,
    train_limit: int,
    test_limit: int,
    shard_minutes: int,
    enable_selection_ablations: bool,
    enable_validation_sensitivity: bool,
    truncate_at_12e: bool = True,
) -> nbformat.NotebookNode:
    template_path = root / "notebooks" / "CoRe_TFM_Q1_FAST_COMPLETE_256_Colab.ipynb"
    template = load_notebook(template_path)
    patched = patch_q1_notebook(
        template,
        run_id=run_id,
        seed=seed,
        train_limit=train_limit,
        test_limit=test_limit,
        drive_base=str(output_root),
        session_minutes=shard_minutes,
        shard_minutes=shard_minutes,
        disable_controlled_replications=True,
        disable_selection_ablations=not enable_selection_ablations,
        disable_validation_sensitivity=not enable_validation_sensitivity,
    )

    first_code = next(i for i, c in enumerate(patched["cells"]) if c.get("cell_type") == "code")
    patched["cells"][first_code]["source"] = _local_setup_source(
        root=root,
        output_root=output_root,
        run_id=run_id,
        seed=seed,
        train_limit=train_limit,
        test_limit=test_limit,
    )

    # Correct provenance text for non-256 context variants without changing benchmark mechanics.
    old = "Bound every fold to 256 training and 128 test rows; use two TabICL estimators; reuse fitted validation predictions for sensitivity."
    new = (
        f"Bound every fold to at most {train_limit} training and {test_limit} test rows; "
        "use two TabICL estimators; reuse fitted validation predictions for sensitivity."
    )
    for cell in patched["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            src = src.replace(old, new)
            cell["source"] = src

    if truncate_at_12e:
        kept = []
        found = False
        for cell in patched["cells"]:
            kept.append(cell)
            if cell.get("cell_type") == "code":
                src = "".join(cell.get("source", []))
                first = src.lstrip().splitlines()[0] if src.strip() else ""
                if "Cell 12E" in first:
                    found = True
                    break
        if not found:
            raise RuntimeError("Could not locate Cell 12E in Q1 template.")
        patched["cells"] = kept

    for cell in patched["cells"]:
        cell["execution_count"] = None
        if cell.get("cell_type") == "code":
            cell["outputs"] = []

    return nbformat.from_dict(patched)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--config", default=None)
    ap.add_argument("--group", choices=["multi_seed", "context", "manual"], default="manual")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-limit", type=int, required=True)
    ap.add_argument("--test-limit", type=int, default=128)
    ap.add_argument("--shard-minutes", type=int, default=120)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--output-notebook", type=Path, default=None)
    ap.add_argument("--disable-ablations", action="store_true")
    ap.add_argument("--disable-validation-sensitivity", action="store_true")
    ap.add_argument("--full-notebook", action="store_true")
    a = ap.parse_args()

    paths = resolve_paths(a.repo, a.config)
    set_low_vram_environment()
    run_id = a.run_id or variant_id(a.group, a.seed, a.train_limit)
    run_dir = paths.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_nb = a.output_notebook or (run_dir / "generated_variant.ipynb")

    nb = build_variant(
        root=paths.repo,
        output_root=paths.output_root,
        run_id=run_id,
        seed=a.seed,
        train_limit=a.train_limit,
        test_limit=a.test_limit,
        shard_minutes=a.shard_minutes,
        enable_selection_ablations=not a.disable_ablations,
        enable_validation_sensitivity=not a.disable_validation_sensitivity,
        truncate_at_12e=not a.full_notebook,
    )
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_nb)
    print(out_nb)


if __name__ == "__main__":
    main()
