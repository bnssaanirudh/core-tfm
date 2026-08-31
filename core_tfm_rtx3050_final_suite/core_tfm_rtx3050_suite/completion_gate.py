from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import json_dump, load_config, resolve_paths, variant_id


def file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_complete(run_dir: Path) -> bool:
    return file_ok(run_dir / "COMPLETE.json") and file_ok(run_dir / "fold_results.csv")


def main():
    ap = argparse.ArgumentParser(description="Paper completion gate for the RTX3050 final suite")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    paths = resolve_paths(args.repo, args.config)
    cfg = load_config(paths.config_path)

    checks = {}
    missing = []
    unsupported = []

    # Five seed robustness runs.
    seed_runs = []
    train = int(cfg["multi_seed"]["train_limit"])
    for seed in cfg["multi_seed"]["seeds"]:
        r = paths.output_root / variant_id("multi_seed", int(seed), train)
        ok = run_complete(r)
        seed_runs.append({"seed": int(seed), "run": str(r), "complete": ok})
        if not ok:
            missing.append(f"multi_seed seed={seed}")
    checks["multi_seed"] = seed_runs

    # Context grid; 256 runs may reuse multi-seed outputs.
    context_runs = []
    multi_seed_set = set(int(x) for x in cfg["multi_seed"]["seeds"])
    for seed in cfg["context_size"]["seeds"]:
        for size in cfg["context_size"]["train_sizes"]:
            seed, size = int(seed), int(size)
            if size == train and seed in multi_seed_set:
                r = paths.output_root / variant_id("multi_seed", seed, size)
                source = "reused_multi_seed"
            else:
                r = paths.output_root / variant_id("context", seed, size)
                source = "context"
            ok = run_complete(r)
            err = r / "EXECUTION_ERROR.json"
            oom = False
            if err.exists():
                try:
                    oom = bool(json.loads(err.read_text(encoding="utf-8")).get("possible_cuda_oom"))
                except Exception:
                    pass
            context_runs.append({"seed": seed, "train_limit": size, "run": str(r), "complete": ok, "source": source, "cuda_oom": oom})
            if not ok:
                if oom and bool(cfg["completion_gate"].get("allow_operationally_unsupported_context_cells", False)):
                    unsupported.append(f"context seed={seed} train={size}: CUDA OOM recorded")
                else:
                    missing.append(f"context seed={seed} train={size}")
    checks["context_size"] = context_runs

    # Analysis artifacts.
    analysis = paths.output_root / "analysis"
    artifacts = {
        "aggregate": analysis / "ANALYSIS_SUMMARY.json",
        "multi_seed_summary": analysis / "multi_seed_summary.json",
        "context_summary": analysis / "context_summary.csv",
        "selection_ablations": analysis / "all_selection_ablations.csv",
        "validation_sensitivity": analysis / "all_validation_fraction_sensitivity.csv",
        "view_reliability": analysis / "view_reliability.csv",
        "view_reliability_summary": analysis / "view_reliability_summary.csv",
        "rare_class_support": analysis / "rare_class" / "dataset_support.csv",
        "rare_class_thresholds": analysis / "rare_class" / "threshold_effects.csv",
        "safe_selective_controlled": paths.output_root / "cpu_experiments" / "safe_selective_controlled" / "COMPLETE.json",
        "pytest_cpu_gate": paths.output_root / "cpu_experiments" / "CPU_EXPERIMENTS_COMPLETE.json",
        "preflight": paths.output_root / "PREFLIGHT.json",
    }
    artifact_status = {k: {"path": str(p), "ok": file_ok(p)} for k, p in artifacts.items()}
    checks["artifacts"] = artifact_status
    for k, s in artifact_status.items():
        if not s["ok"]:
            missing.append(k)

    # Sanity-read key CSVs so header-only files cannot pass.
    for key in ["selection_ablations", "validation_sensitivity", "view_reliability", "rare_class_thresholds"]:
        p = artifacts[key]
        if file_ok(p):
            try:
                if pd.read_csv(p).empty:
                    missing.append(f"{key}: empty CSV")
            except Exception as exc:
                missing.append(f"{key}: parse error {exc}")

    complete = len(missing) == 0
    payload = {
        "complete": complete,
        "missing": sorted(set(missing)),
        "operationally_unsupported_but_recorded": unsupported,
        "checks": checks,
        "interpretation": (
            "PASS means all required final evidence groups are present. Context cells explicitly recorded as CUDA OOM may be listed as operationally unsupported rather than silently omitted."
        ),
    }
    json_dump(paths.output_root / "FINAL_COMPLETION_GATE.json", payload)
    print(json.dumps(payload, indent=2))
    if not complete:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
