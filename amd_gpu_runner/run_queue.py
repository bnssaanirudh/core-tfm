from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from core_tfm.robustness_runner import fold_result_status


def preflight_ok(root: Path) -> bool:
    p = root / "results" / "amd_preflight" / "amd_gpu_preflight.json"
    if not p.exists():
        return False
    try:
        x = json.loads(p.read_text(encoding="utf-8"))
        return all(bool(x.get("models", {}).get(m, {}).get("ok")) for m in ("tabiclv2", "tabpfn3"))
    except Exception:
        return False


def queue_from_config(root: Path) -> tuple[list[dict], dict]:
    cfg = yaml.safe_load((root / "configs" / "reliability_aware_experiments.yaml").read_text(encoding="utf-8"))
    q: list[dict] = []
    for seed in cfg["seed_robustness"]["seeds"]:
        q.append({
            "group": "multi_seed",
            "name": f"seed_{seed}",
            "run_id": f"multi_seed/seed_{seed}",
            "seed": int(seed),
            "train_limit": int(cfg["seed_robustness"]["train_limit"]),
            "test_limit": int(cfg["seed_robustness"]["test_limit"]),
        })
    for seed in cfg["context_size"]["seeds"]:
        for size in cfg["context_size"]["train_sizes"]:
            q.append({
                "group": "context_size",
                "name": f"seed_{seed}_train_{size}",
                "run_id": f"context_size/seed_{seed}_train_{size}",
                "seed": int(seed),
                "train_limit": int(size),
                "test_limit": int(cfg["context_size"].get("test_limit", 128)),
            })
    return q, cfg


def summarize(root: Path, output_root: Path, cfg: dict) -> None:
    script = root / "experiments" / "summarize_robustness_runs.py"
    cmd = [
        sys.executable, str(script),
        "--root", str(output_root),
        "--seeds", ",".join(map(str, cfg["seed_robustness"]["seeds"])),
        "--context-seeds", ",".join(map(str, cfg["context_size"]["seeds"])),
        "--context-sizes", ",".join(map(str, cfg["context_size"]["train_sizes"])),
    ]
    subprocess.run(cmd, cwd=root, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("results/amd_jmlr_robustness_v1"))
    ap.add_argument("--shard-minutes", type=int, default=45)
    ap.add_argument("--kernel", default="core-tfm-amd")
    ap.add_argument("--max-variants", type=int, default=1, help="How many incomplete variants to attempt in this invocation.")
    ap.add_argument("--all", action="store_true", help="Run every incomplete variant sequentially.")
    ap.add_argument("--group", choices=["multi_seed", "context_size", "all"], default="all")
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = (root / a.output_root).resolve() if not a.output_root.is_absolute() else a.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not preflight_ok(root):
        raise SystemExit("AMD model preflight has not passed. Run: python -m amd_gpu_runner.preflight")

    queue, cfg = queue_from_config(root)
    if a.group != "all":
        queue = [v for v in queue if v["group"] == a.group]

    pending = []
    rows = []
    for v in queue:
        st = fold_result_status(output_root / v["run_id"])
        rows.append({
            **v,
            "complete": bool(st.get("complete")),
            "rows": int(st.get("rows", 0)),
            "fold_cells": int(st.get("fold_cells", 0)),
            "failure_count": int(st.get("failure_count", 0)),
        })
        if not st.get("complete"):
            pending.append(v)

    (output_root / "QUEUE_STATUS.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Queue: {len(queue) - len(pending)}/{len(queue)} variants complete; {len(pending)} pending.")
    if not pending:
        summarize(root, output_root, cfg)
        print("Requested queue is complete.")
        return

    limit = len(pending) if a.all else max(1, a.max_variants)
    for i, v in enumerate(pending[:limit], 1):
        print(f"\n=== AMD VARIANT {i}/{min(limit, len(pending))}: {v['run_id']} ===")
        cmd = [
            sys.executable, "-m", "amd_gpu_runner.run_variant",
            "--run-id", v["run_id"],
            "--seed", str(v["seed"]),
            "--train-limit", str(v["train_limit"]),
            "--test-limit", str(v["test_limit"]),
            "--shard-minutes", str(a.shard_minutes),
            "--output-root", str(output_root),
            "--kernel", a.kernel,
        ]
        result = subprocess.run(cmd, cwd=root)
        if result.returncode != 0:
            print("Variant exited with an error. Its completed fold checkpoints are preserved.")
            print("Fix the reported compatibility/data error, then rerun the same queue command.")
            break
        summarize(root, output_root, cfg)

    # Always write final current queue state.
    final = []
    for v in queue:
        st = fold_result_status(output_root / v["run_id"])
        final.append({**v, **st})
    (output_root / "QUEUE_STATUS.json").write_text(json.dumps(final, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
