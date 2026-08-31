from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .common import json_dump, load_config, resolve_paths, set_low_vram_environment, variant_id
from .run_variant import execute_variant


def planned_variants(cfg: dict, profile: str):
    variants = []
    if profile in {"paper", "multi_seed"}:
        train = int(cfg["multi_seed"]["train_limit"])
        for seed in cfg["multi_seed"]["seeds"]:
            variants.append(("multi_seed", int(seed), train))
    if profile in {"paper", "context"}:
        seen = {(s, t) for _, s, t in variants}
        for seed in cfg["context_size"]["seeds"]:
            for train in cfg["context_size"]["train_sizes"]:
                key = (int(seed), int(train))
                if key in seen:
                    continue
                variants.append(("context", key[0], key[1]))
                seen.add(key)
    return variants


def main():
    ap = argparse.ArgumentParser(description="Run/resume the final CoRe-TFM robustness matrix")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", choices=["paper", "multi_seed", "context"], default="paper")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--max-variants", type=int, default=None)
    args = ap.parse_args()

    paths = resolve_paths(args.repo, args.config)
    cfg = load_config(paths.config_path)
    set_low_vram_environment()

    variants = planned_variants(cfg, args.profile)
    if args.max_variants is not None:
        variants = variants[: args.max_variants]

    state = {"profile": args.profile, "planned": len(variants), "variants": []}
    test_limit = int(cfg["benchmark"]["test_limit"])
    shard_minutes = int(cfg["benchmark"]["shard_minutes"])
    ablations = bool(cfg["benchmark"]["enable_selection_ablations"])
    sensitivity = bool(cfg["benchmark"]["enable_validation_sensitivity"])

    for idx, (group, seed, train) in enumerate(variants, 1):
        run_id = variant_id(group, seed, train)
        print(f"\n=== [{idx}/{len(variants)}] {run_id} ===", flush=True)
        try:
            result = execute_variant(
                repo=paths.repo,
                output_root=paths.output_root,
                run_id=run_id,
                seed=seed,
                train_limit=train,
                test_limit=test_limit,
                shard_minutes=shard_minutes,
                enable_selection_ablations=ablations,
                enable_validation_sensitivity=sensitivity,
            )
        except Exception as exc:  # noqa: BLE001
            result = {"run_id": run_id, "status": "runner_error", "error": repr(exc)}
            if args.stop_on_error:
                state["variants"].append(result)
                json_dump(paths.output_root / "MATRIX_STATUS.json", state)
                raise
        state["variants"].append(result)
        json_dump(paths.output_root / "MATRIX_STATUS.json", state)

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    state["complete_variants"] = sum(v.get("status") in {"complete", "already_complete"} for v in state["variants"])
    state["incomplete_variants"] = len(state["variants"]) - state["complete_variants"]
    json_dump(paths.output_root / "MATRIX_STATUS.json", state)
    print(json.dumps(state, indent=2, default=str))


if __name__ == "__main__":
    main()
