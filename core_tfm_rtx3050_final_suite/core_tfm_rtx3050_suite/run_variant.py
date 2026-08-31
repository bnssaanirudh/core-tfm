from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from core_tfm.robustness_runner import fold_result_status, write_complete_marker

from .common import git_head, json_dump, resolve_paths, set_low_vram_environment, variant_id
from .make_variant import build_variant


def execute_variant(
    *,
    repo: Path,
    output_root: Path,
    run_id: str,
    seed: int,
    train_limit: int,
    test_limit: int,
    shard_minutes: int,
    enable_selection_ablations: bool,
    enable_validation_sensitivity: bool,
) -> dict:
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    generated = run_dir / "generated_variant.ipynb"
    executed = run_dir / "executed_variant.ipynb"

    before = fold_result_status(run_dir)
    if before.get("complete"):
        return {"status": "already_complete", "run_id": run_id, "fold_result_status": before}

    nb = build_variant(
        root=repo,
        output_root=output_root,
        run_id=run_id,
        seed=seed,
        train_limit=train_limit,
        test_limit=test_limit,
        shard_minutes=shard_minutes,
        enable_selection_ablations=enable_selection_ablations,
        enable_validation_sensitivity=enable_validation_sensitivity,
        truncate_at_12e=True,
    )
    # Normalize notebook cell sources for nbclient compatibility.
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            if isinstance(source, list):
                cell["source"] = "".join(source)
    nbformat.write(nb, generated)

    

    start = time.time()
    error = None
    try:
        client = NotebookClient(
            nb,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(repo)}},
            allow_errors=False,
            record_timing=True,
        )
        client.execute()
    except Exception as exc:  # noqa: BLE001
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "possible_cuda_oom": "out of memory" in str(exc).lower(),
        }
        json_dump(run_dir / "EXECUTION_ERROR.json", error)
    finally:
        nbformat.write(nb, executed)

    after = fold_result_status(run_dir)
    payload = {
        "run_id": run_id,
        "seed": seed,
        "train_limit": train_limit,
        "test_limit": test_limit,
        "elapsed_seconds": time.time() - start,
        "source_commit": git_head(repo),
        "fold_result_status": after,
        "error": error,
    }

    # Selection ablation/sensitivity rows are produced from the same fitted views.
    for name in ["selection_ablations.csv", "validation_fraction_sensitivity.csv"]:
        p = run_dir / name
        payload[name] = {"exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0}

    json_dump(run_dir / "RUN_STATUS.json", payload)

    if after.get("complete"):
        marker = run_dir / "COMPLETE.json"
        if not marker.exists():
            write_complete_marker(
                run_dir,
                {
                    "execution_platform": "windows_local_nvidia_cuda",
                    "source_commit": git_head(repo),
                    "seed": seed,
                    "requested_train_limit": train_limit,
                    "requested_test_limit": test_limit,
                    "executed_notebook": executed.name,
                    "selection_ablations_enabled": enable_selection_ablations,
                    "validation_sensitivity_enabled": enable_validation_sensitivity,
                },
            )
        payload["status"] = "complete"
    else:
        payload["status"] = "incomplete"
    return payload


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
    ap.add_argument("--disable-ablations", action="store_true")
    ap.add_argument("--disable-validation-sensitivity", action="store_true")
    args = ap.parse_args()

    paths = resolve_paths(args.repo, args.config)
    set_low_vram_environment()
    run_id = args.run_id or variant_id(args.group, args.seed, args.train_limit)
    payload = execute_variant(
        repo=paths.repo,
        output_root=paths.output_root,
        run_id=run_id,
        seed=args.seed,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        shard_minutes=args.shard_minutes,
        enable_selection_ablations=not args.disable_ablations,
        enable_validation_sensitivity=not args.disable_validation_sensitivity,
    )
    print(json.dumps(payload, indent=2, default=str))
    if payload.get("status") == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
