from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from core_tfm.robustness_runner import fold_result_status, write_complete_marker
from amd_gpu_runner.make_local_notebook import build_notebook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-limit", type=int, required=True)
    ap.add_argument("--test-limit", type=int, default=128)
    ap.add_argument("--shard-minutes", type=int, default=45)
    ap.add_argument("--output-root", type=Path, default=Path("results/amd_jmlr_robustness_v1"))
    ap.add_argument("--kernel", default="core-tfm-amd")
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = (root / a.output_root).resolve() if not a.output_root.is_absolute() else a.output_root.resolve()
    run_dir = output_root / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if not token:
        raise SystemExit("TABPFN_TOKEN is not set.")

    before = fold_result_status(run_dir)
    print("BEFORE:", json.dumps(before, indent=2))
    if before.get("complete"):
        if not (run_dir / "COMPLETE.json").exists():
            write_complete_marker(run_dir, {
                "execution_platform": "local_linux_amd_rocm",
                "seed": a.seed,
                "requested_train_limit": a.train_limit,
                "requested_test_limit": a.test_limit,
            })
        print("Variant already complete; nothing to do.")
        return

    nb = build_notebook(
        root=root,
        output_root=output_root,
        run_id=a.run_id,
        seed=a.seed,
        train_limit=a.train_limit,
        test_limit=a.test_limit,
        shard_minutes=a.shard_minutes,
    )
    generated = run_dir / "generated_amd_variant.ipynb"
    executed = run_dir / "executed_amd_variant.ipynb"
    nbformat.write(nb, generated)
    print("Generated notebook:", generated)

    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name=a.kernel,
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,
        record_timing=True,
    )

    try:
        client.execute()
    except Exception as exc:
        nbformat.write(nb, executed)
        error_payload = {
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "run_id": a.run_id,
            "seed": a.seed,
            "train_limit": a.train_limit,
            "test_limit": a.test_limit,
        }
        (run_dir / "EXECUTION_ERROR.json").write_text(json.dumps(error_payload, indent=2), encoding="utf-8")
        print("Execution failed; partial fold checkpoints were preserved.")
        print("Executed notebook:", executed)
        print(json.dumps(error_payload, indent=2))
        raise
    else:
        nbformat.write(nb, executed)
        error_file = run_dir / "EXECUTION_ERROR.json"
        if error_file.exists():
            error_file.unlink()

    after = fold_result_status(run_dir)
    print("AFTER:", json.dumps(after, indent=2))
    if after.get("complete"):
        marker = write_complete_marker(run_dir, {
            "execution_platform": "local_linux_amd_rocm",
            "seed": a.seed,
            "requested_train_limit": a.train_limit,
            "requested_test_limit": a.test_limit,
            "generated_notebook": generated.name,
            "executed_notebook": executed.name,
        })
        print("VARIANT COMPLETE:", marker)
    else:
        print("Variant remains incomplete. Run the same command again; completed folds will be skipped.")
        if after.get("last_failure"):
            print("LAST FOLD FAILURE:")
            print(json.dumps(after["last_failure"], indent=2))


if __name__ == "__main__":
    main()
