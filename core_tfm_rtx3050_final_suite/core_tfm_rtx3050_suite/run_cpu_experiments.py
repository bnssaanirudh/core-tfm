from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import json_dump, load_config, resolve_paths


def run(cmd, cwd: Path):
    print("RUN:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser(description="CPU-side experiments and audits that accompany the GPU matrix")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    paths = resolve_paths(args.repo, args.config)
    cfg = load_config(paths.config_path)
    out = paths.output_root / "cpu_experiments"
    out.mkdir(parents=True, exist_ok=True)

    # Repository tests first: catches reconciliation/metric regressions before expensive inference.
    run([sys.executable, "-m", "pytest", "-q"], paths.repo)

    safe_out = out / "safe_selective_controlled"
    c = cfg["controlled_safe"]
    run([
        sys.executable,
        str(paths.repo / "experiments" / "run_safe_selective_controlled.py"),
        "--tasks", str(int(c["tasks"])),
        "--n", str(int(c["n"])),
        "--beta", str(float(c["beta"])),
        "--delta", str(float(c["delta"])),
        "--output-dir", str(safe_out),
    ], paths.repo)

    # Rebuild the frozen archive-derived reliability story as a sanity/reference analysis.
    frozen = paths.repo / "results" / "q1_fast_complete_256_v1" / "fold_results.csv"
    archive_out = out / "frozen_reliability_reference"
    if frozen.exists():
        run([
            sys.executable,
            str(paths.repo / "experiments" / "run_reliability_aware_suite.py"),
            "--fold-results", str(frozen),
            "--output", str(archive_out),
        ], paths.repo)

    payload = {
        "pytest": "PASS",
        "safe_selective_controlled_complete": (safe_out / "COMPLETE.json").exists(),
        "safe_selective_summary": str(safe_out / "summary.json"),
        "frozen_reference_generated": (archive_out / "RUN_METADATA.json").exists(),
    }
    json_dump(out / "CPU_EXPERIMENTS_COMPLETE.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
