from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

from core_tfm.robustness_runner import load_notebook, patch_q1_notebook


def _local_setup_source(
    *, root: Path, output_root: Path, run_id: str, seed: int, train_limit: int, test_limit: int
) -> str:
    protocol_revision = f"amd_rocm_seed{seed}_train{train_limit}_test{test_limit}"
    return f'''# Local AMD/ROCm setup replacing the Colab-only Cell 2.
FULL_Q1_RUN = True
RUN_ID = {run_id!r}
PROTOCOL_REVISION = {protocol_revision!r}

from pathlib import Path
import importlib
import importlib.metadata
import json
import os
import shutil
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
            "Checkout the frozen commit or use a new output root."
        )
else:
    frozen = {{"core_tfm_commit": head}}

frozen.update({{
    "protocol_revision": PROTOCOL_REVISION,
    "execution_platform": "local_linux_amd_rocm",
    "bounded_context_protocol": {{"max_train_rows": {train_limit}, "max_test_rows": {test_limit}}},
    "catboost_version": "1.2.10",
}})
FREEZE_PATH.write_text(json.dumps(frozen, indent=2), encoding="utf-8")

import torch
if torch.version.hip is None:
    raise RuntimeError("This run requires a ROCm/HIP PyTorch build.")
if not torch.cuda.is_available():
    raise RuntimeError("ROCm GPU is not visible to PyTorch.")
print("ROCm/HIP:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))

# TabPFN headless authentication. Model execution remains the real compatibility test.
tabpfn_token = os.environ.get("TABPFN_TOKEN", "").strip()
if not tabpfn_token:
    raise RuntimeError("Set TABPFN_TOKEN before running the AMD benchmark.")
os.environ["TABPFN_TOKEN"] = tabpfn_token
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
            raise RuntimeError("Accept the TabPFN-3 license in your Prior Labs account, then rerun.") from exc
        raise
except ImportError:
    # API layout can vary; TabPFNClassifier will still validate/download on first use.
    pass

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)
print("Persistent run directory:", RUN)
print("Protocol revision:", PROTOCOL_REVISION)
print("Frozen source:", json.dumps(frozen, indent=2))
'''


def build_notebook(
    *, root: Path, output_root: Path, run_id: str, seed: int, train_limit: int,
    test_limit: int, shard_minutes: int
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
        disable_selection_ablations=True,
        disable_validation_sensitivity=True,
    )

    # Replace only the first code cell (the Colab-specific setup). All benchmark
    # data/model/evaluator/shard cells remain from the original Q1 notebook.
    first_code = next(i for i, c in enumerate(patched["cells"]) if c.get("cell_type") == "code")
    patched["cells"][first_code]["source"] = _local_setup_source(
        root=root,
        output_root=output_root,
        run_id=run_id,
        seed=seed,
        train_limit=train_limit,
        test_limit=test_limit,
    ).splitlines(keepends=True)

    # Correct the human-readable amendment text for non-256 context variants.
    for cell in patched["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        src = src.replace(
            "Bound every fold to 256 training and 128 test rows; use two TabICL estimators; reuse fitted validation predictions for sensitivity.",
            f"Bound every fold to at most {train_limit} training and {test_limit} test rows; use two TabICL estimators; disable unrelated post-processing for this robustness variant.",
        )
        cell["source"] = src.splitlines(keepends=True)

    # Keep cells only through 12E. This avoids Q1 manuscript/controlled-study
    # post-processing and makes each robustness variant a pure inference run.
    kept = []
    found_12e = False
    for cell in patched["cells"]:
        kept.append(cell)
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            first = src.lstrip().splitlines()[0] if src.strip() else ""
            if "Cell 12E" in first:
                found_12e = True
                break
    if not found_12e:
        raise RuntimeError("Could not find Cell 12E in Q1 template.")
    patched["cells"] = kept

    # Clear stale execution artifacts from the template.
    for cell in patched["cells"]:
        cell["execution_count"] = None
        if cell.get("cell_type") == "code":
            cell["outputs"] = []

    return nbformat.from_dict(patched)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-limit", type=int, required=True)
    ap.add_argument("--test-limit", type=int, default=128)
    ap.add_argument("--shard-minutes", type=int, default=120)
    ap.add_argument("--output-root", type=Path, default=Path("results/amd_jmlr_robustness_v1"))
    ap.add_argument("--output-notebook", type=Path, required=True)
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = (root / a.output_root).resolve() if not a.output_root.is_absolute() else a.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    nb = build_notebook(
        root=root,
        output_root=output_root,
        run_id=a.run_id,
        seed=a.seed,
        train_limit=a.train_limit,
        test_limit=a.test_limit,
        shard_minutes=a.shard_minutes,
    )
    a.output_notebook.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, a.output_notebook)
    print(a.output_notebook)


if __name__ == "__main__":
    main()
