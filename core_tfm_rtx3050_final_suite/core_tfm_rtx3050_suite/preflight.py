from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .common import git_head, json_dump, resolve_paths, set_low_vram_environment


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="RTX 3050 / Windows preflight for CoRe-TFM final suite")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--config", default=None)
    ap.add_argument("--strict-token", action="store_true")
    args = ap.parse_args()

    paths = resolve_paths(args.repo, args.config)
    set_low_vram_environment()

    import torch

    report = {
        "repo": str(paths.repo),
        "source_commit": git_head(paths.repo),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "tabpfn_token_present": bool(os.environ.get("TABPFN_TOKEN", "").strip()),
        "packages": {},
    }

    for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "openml", "catboost", "tabicl", "tabpfn", "nbformat", "nbclient", "pyyaml"]:
        report["packages"][pkg] = package_version(pkg)

    if not torch.cuda.is_available():
        report["error"] = "PyTorch cannot see a CUDA GPU."
        json_dump(paths.output_root / "PREFLIGHT.json", report)
        raise RuntimeError(
            "CUDA is not available to PyTorch. Install the CUDA-enabled PyTorch build and update the NVIDIA driver."
        )

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    report.update({
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_vram_gb": round(total_gb, 3),
        "gpu_free_vram_gb_at_preflight": round(free_bytes / 1024**3, 3),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
    })

    # Tiny CUDA computation: catches broken driver/runtime combinations early.
    x = torch.randn((1024, 1024), device="cuda")
    y = x @ x.T
    checksum = float(y[0, 0].detach().cpu())
    del x, y
    torch.cuda.empty_cache()
    report["cuda_matmul_checksum"] = checksum

    required_imports = ["core_tfm", "tabicl", "tabpfn", "catboost", "nbformat", "nbclient"]
    import_failures = {}
    for mod in required_imports:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            import_failures[mod] = repr(exc)
    report["import_failures"] = import_failures

    if args.strict_token and not report["tabpfn_token_present"]:
        report["error"] = "TABPFN_TOKEN is missing."
        json_dump(paths.output_root / "PREFLIGHT.json", report)
        raise RuntimeError("TABPFN_TOKEN is missing from this PowerShell session.")

    if import_failures:
        report["error"] = "One or more required imports failed."
        json_dump(paths.output_root / "PREFLIGHT.json", report)
        raise RuntimeError(f"Import failures: {json.dumps(import_failures, indent=2)}")

    # Record nvidia-smi if available.
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        report["nvidia_smi"] = smi
    except Exception as exc:  # noqa: BLE001
        report["nvidia_smi_error"] = repr(exc)

    report["status"] = "PASS"
    json_dump(paths.output_root / "PREFLIGHT.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
