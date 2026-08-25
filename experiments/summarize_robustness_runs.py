"""Aggregate completed CoRe-TFM robustness variants and write evidence gates.

The script is intentionally safe on partial progress: group directories are
created before any CSV/JSON writes, empty summaries get stable schemas, and a
missing/incomplete variant is reported rather than treated as an exception.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRIMARY = ("tabiclv2", "tabpfn3")
METHODS = 8
DATASETS = 10
FOLDS = 5
MODELS = 3
EXPECTED_ROWS = DATASETS * FOLDS * MODELS * METHODS

SEED_SUMMARY_COLUMNS = [
    "seed", "mean_selective_minus_arithmetic", "selective_wins", "n_datasets",
    "wilcoxon_p", "mean_train_rows", "min_train_rows", "max_train_rows",
    "mean_validation_rows", "mean_test_rows",
]
CONTEXT_SUMMARY_COLUMNS = [
    "seed", "requested_train_size", "mean_selective_minus_arithmetic",
    "selective_wins", "n_datasets", "wilcoxon_p", "mean_train_rows",
    "min_train_rows", "max_train_rows", "mean_validation_rows", "mean_test_rows",
]


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def check_variant(path: Path) -> dict:
    f = path / "fold_results.csv"
    df = _safe_read_csv(f)
    if df is None:
        return {"complete": False, "reason": "fold_results missing/empty/unparseable", "path": str(path), "rows": 0}
    required = {"dataset", "model", "fold", "method", "joint_nll", "n_train", "n_validation", "n_test"}
    missing = required - set(df.columns)
    if missing:
        return {"complete": False, "reason": f"missing columns {sorted(missing)}", "path": str(path), "rows": int(len(df))}
    if df.empty:
        return {"complete": False, "reason": "fold_results has zero data rows", "path": str(path), "rows": 0}
    per = df.groupby(["dataset", "model", "fold"])["method"].nunique()
    ok = (
        len(df) == EXPECTED_ROWS
        and df["dataset"].nunique() == DATASETS
        and df["model"].nunique() == MODELS
        and df[["dataset", "model", "fold"]].drop_duplicates().shape[0] == DATASETS * MODELS * FOLDS
        and int(per.min()) == METHODS
        and int(per.max()) == METHODS
    )
    return {
        "complete": bool(ok), "path": str(path), "rows": int(len(df)),
        "datasets": int(df["dataset"].nunique()), "models": int(df["model"].nunique()),
        "fold_cells": int(df[["dataset", "model", "fold"]].drop_duplicates().shape[0]),
        "methods_per_fold_min": int(per.min()), "methods_per_fold_max": int(per.max()),
    }


def primary_dataset_effect(df: pd.DataFrame) -> pd.Series:
    x = df[df["model"].isin(PRIMARY)]
    wide = x.pivot_table(index=["dataset", "model", "fold"], columns="method", values="joint_nll", aggfunc="first")
    required = {"selective_core", "arithmetic"}
    if not required <= set(wide.columns):
        raise ValueError(f"Missing primary methods: {sorted(required - set(wide.columns))}")
    d = (wide["selective_core"] - wide["arithmetic"]).groupby(["dataset", "model"]).mean()
    return d.groupby("dataset").mean().sort_index()


def summarize_variant(path: Path, **tags) -> dict:
    df = pd.read_csv(path / "fold_results.csv")
    effect = primary_dataset_effect(df)
    values = effect.to_numpy()
    p = float(wilcoxon(values).pvalue) if np.any(values != 0) else 1.0
    row = {
        **tags,
        "mean_selective_minus_arithmetic": float(effect.mean()),
        "selective_wins": int((effect < 0).sum()),
        "n_datasets": int(len(effect)),
        "wilcoxon_p": p,
        "mean_train_rows": float(df["n_train"].mean()),
        "min_train_rows": int(df["n_train"].min()),
        "max_train_rows": int(df["n_train"].max()),
        "mean_validation_rows": float(df["n_validation"].mean()),
        "mean_test_rows": float(df["n_test"].mean()),
    }
    for ds, val in effect.items():
        row[f"effect_{ds}"] = float(val)
    return row


def aggregate_multi_seed(root: Path, seeds: list[int]) -> dict:
    group = root / "multi_seed"
    group.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, dict] = {}
    rows: list[dict] = []
    for seed in seeds:
        p = group / f"seed_{seed}"
        s = check_variant(p)
        statuses[str(seed)] = s
        if s["complete"]:
            rows.append(summarize_variant(p, seed=seed))
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=SEED_SUMMARY_COLUMNS)
    frame.to_csv(group / "summary_by_seed.csv", index=False)
    complete = len(rows) == len(seeds)
    payload = {"complete": complete, "expected_seeds": seeds, "completed_variants": len(rows), "statuses": statuses}
    (group / "STATUS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    complete_path = group / "COMPLETE.json"
    if complete:
        effects = pd.DataFrame(rows)["mean_selective_minus_arithmetic"].to_numpy()
        payload["mean_across_seed_effects"] = float(effects.mean())
        payload["sd_across_seed_effects"] = float(effects.std(ddof=1)) if len(effects) > 1 else 0.0
        complete_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif complete_path.exists():
        complete_path.unlink()
    return payload


def aggregate_context(root: Path, seeds: list[int], sizes: list[int]) -> dict:
    group = root / "context_size"
    group.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, dict] = {}
    rows: list[dict] = []
    for seed in seeds:
        for size in sizes:
            key = f"seed_{seed}_train_{size}"
            p = group / key
            s = check_variant(p)
            statuses[key] = s
            if s["complete"]:
                rows.append(summarize_variant(p, seed=seed, requested_train_size=size))
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=CONTEXT_SUMMARY_COLUMNS)
    frame.to_csv(group / "summary_by_seed_context.csv", index=False)
    complete = len(rows) == len(seeds) * len(sizes)
    payload = {
        "complete": complete, "expected_seeds": seeds, "expected_sizes": sizes,
        "completed_variants": len(rows), "expected_variants": len(seeds) * len(sizes),
        "statuses": statuses,
    }
    (group / "STATUS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    complete_path = group / "COMPLETE.json"
    if complete:
        by = pd.DataFrame(rows).groupby("requested_train_size", as_index=False).agg(
            mean_effect=("mean_selective_minus_arithmetic", "mean"),
            sd_effect=("mean_selective_minus_arithmetic", "std"),
            mean_actual_train=("mean_train_rows", "mean"),
        )
        by.to_csv(group / "summary_by_context.csv", index=False)
        complete_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif complete_path.exists():
        complete_path.unlink()
    return payload


def rare_class_from_multiseed(root: Path, seeds: list[int], support_map: dict[str, int], thresholds: list[int]) -> dict:
    out = root / "rare_class"
    out.mkdir(parents=True, exist_ok=True)
    seed_root = root / "multi_seed"
    if not all(check_variant(seed_root / f"seed_{s}")["complete"] for s in seeds):
        payload = {"complete": False, "reason": "multi_seed must complete first"}
        (out / "STATUS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        complete_path = out / "COMPLETE.json"
        if complete_path.exists():
            complete_path.unlink()
        return payload
    rows = []
    for seed in seeds:
        df = pd.read_csv(seed_root / f"seed_{seed}" / "fold_results.csv")
        for threshold in thresholds:
            keep = [d for d, n in support_map.items() if n >= threshold]
            keep += [d for d in sorted(df["dataset"].unique()) if d not in support_map]
            keep = sorted(set(keep))
            sub = df[df["dataset"].isin(keep)]
            effect = primary_dataset_effect(sub)
            if effect.empty:
                continue
            rows.append({
                "seed": seed, "minimum_support_threshold": threshold,
                "n_datasets": int(len(effect)), "datasets": ";".join(effect.index),
                "mean_selective_minus_arithmetic": float(effect.mean()),
                "selective_wins": int((effect < 0).sum()),
            })
    pd.DataFrame(rows).to_csv(out / "rare_class_exclusion_sensitivity.csv", index=False)
    expected = len(seeds) * len(thresholds)
    complete = len(rows) == expected
    payload = {
        "complete": complete, "scope": "dataset-exclusion sensitivity",
        "support_map": support_map, "thresholds": thresholds,
        "rows": len(rows), "expected_rows": expected,
        "note": "Support-adaptive penalty tuning requires fresh candidate/view archives and is not claimed by this exclusion analysis.",
    }
    (out / "STATUS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    complete_path = out / "COMPLETE.json"
    if complete:
        complete_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif complete_path.exists():
        complete_path.unlink()
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seeds", default="11,23,42,71,101")
    ap.add_argument("--context-seeds", default="23,42,71")
    ap.add_argument("--context-sizes", default="64,128,256,512,1024")
    args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    cseeds = [int(x) for x in args.context_seeds.split(",") if x]
    sizes = [int(x) for x in args.context_sizes.split(",") if x]
    support_map = {"customer": 3, "marketing": 2, "nursery": 2}
    report = {
        "multi_seed": aggregate_multi_seed(args.root, seeds),
        "context_size": aggregate_context(args.root, cseeds, sizes),
        "rare_class": rare_class_from_multiseed(args.root, seeds, support_map, [2, 5, 10]),
    }
    (args.root / "ROBUSTNESS_STATUS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v.get("complete") for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
