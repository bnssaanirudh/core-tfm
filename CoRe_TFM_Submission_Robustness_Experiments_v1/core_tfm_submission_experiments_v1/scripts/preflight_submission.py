from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

REQUIRED = [
    "core_tfm_rtx3050_final_suite",
    "experiments/audit_evidence_package.py",
    "experiments/run_reliability_aware_suite.py",
    "notebooks/CoRe_TFM_Q1_FAST_COMPLETE_256_Colab.ipynb",
    "results/q1_fast_complete_256_v1",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    missing = [p for p in REQUIRED if not (repo / p).exists()]
    payload = {
        "repo": str(repo),
        "missing_required_paths": missing,
        "python": sys.version,
        "tabpfn_token_present": bool(os.environ.get("TABPFN_TOKEN", "").strip()),
        "git_head": None,
        "cuda_available": None,
        "gpu": None,
        "frozen_results_read_only_policy": True,
    }
    try:
        payload["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception as e:
        payload["git_error"] = repr(e)

    try:
        import torch
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            payload["gpu"] = torch.cuda.get_device_name(0)
            payload["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 3
            )
    except Exception as e:
        payload["torch_error"] = repr(e)

    print(json.dumps(payload, indent=2))
    if missing:
        raise SystemExit("Repository is missing required CoRe-TFM paths.")
    if not payload["tabpfn_token_present"]:
        print("\nWARNING: TABPFN_TOKEN is not set. GPU TFM runs will not start.")
    print("\nPreflight structure check complete. No experiment was executed.")

if __name__ == "__main__":
    main()
