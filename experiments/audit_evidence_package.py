"""Audit the frozen CoRe-TFM Q1 evidence package.

This script is read-only. It verifies benchmark completeness and recomputes the
headline dataset-blocked statistics used by the manuscripts. It is intentionally
separate from the fresh robustness experiments so that the frozen evidence is not
modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRIMARY_MODELS = ("tabiclv2", "tabpfn3")
EXPECTED_DATASETS = 10
EXPECTED_FOLDS = 5
EXPECTED_MODELS = 3
EXPECTED_METHODS = 8
EXPECTED_ROWS = EXPECTED_DATASETS * EXPECTED_FOLDS * EXPECTED_MODELS * EXPECTED_METHODS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_mean_ci(values: np.ndarray, seed: int = 42, n_boot: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n = len(values)
    for i in range(n_boot):
        boot[i] = rng.choice(values, size=n, replace=True).mean()
    return tuple(np.quantile(boot, [0.025, 0.975]))


def dataset_blocked_effect(df: pd.DataFrame, metric: str) -> pd.Series:
    p = df[df["model"].isin(PRIMARY_MODELS)].copy()
    wide = p.pivot_table(index=["dataset", "model", "fold"], columns="method", values=metric, aggfunc="first")
    effect = (wide["selective_core"] - wide["arithmetic"]).groupby(["dataset", "model"]).mean()
    return effect.groupby("dataset").mean().sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/q1_fast_complete_256_v1"))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    root = args.results

    required = [
        "fold_results.csv", "cell_summary.csv", "dataset_blocked_primary_inference.json",
        "quality_gate.json", "q1_evidence_gate.json", "protocol.json",
        "protocol_amendment.json", "hardware_and_model_environment.json",
        "frozen_source_revisions.json", "sha256sums_final.json",
        "selection_ablations.csv", "validation_fraction_sensitivity.csv",
        "controlled_replications/exact_view_summary.csv",
        "controlled_replications/exact_view_tests.csv",
        "controlled_replications/selective_mixture_summary.csv",
        "controlled_replications/selective_mixture_test.csv",
        "controlled_replications/validation_size_summary.csv",
    ]
    missing = [x for x in required if not (root / x).exists()]

    df = pd.read_csv(root / "fold_results.csv")
    row_count_ok = len(df) == EXPECTED_ROWS
    cell_count = df[["dataset", "model"]].drop_duplicates().shape[0]
    fold_count = df[["dataset", "model", "fold"]].drop_duplicates().shape[0]
    methods_per_fold = df.groupby(["dataset", "model", "fold"])["method"].nunique()

    nll = dataset_blocked_effect(df, "joint_nll")
    nll_ci = bootstrap_mean_ci(nll.to_numpy())
    nll_p = float(wilcoxon(nll.to_numpy(), alternative="two-sided").pvalue)

    metrics = {}
    for metric in ["joint_nll", "joint_brier", "ece", "marginal_distortion"]:
        if metric in df.columns:
            eff = dataset_blocked_effect(df, metric)
            metrics[metric] = {
                "mean_selective_minus_arithmetic": float(eff.mean()),
                "wins": int((eff < 0).sum()),
                "wilcoxon_two_sided_p": float(wilcoxon(eff.to_numpy(), alternative="two-sided").pvalue),
                "dataset_effects": {k: float(v) for k, v in eff.items()},
            }

    # Model-specific NLL reversal.
    p = df[df["model"].isin(PRIMARY_MODELS)]
    wide = p.pivot_table(index=["dataset", "model", "fold"], columns="method", values="joint_nll", aggfunc="first")
    cell = (wide["selective_core"] - wide["arithmetic"]).groupby(["dataset", "model"]).mean()
    model_means = cell.groupby("model").mean()
    paired = cell.unstack("model")
    model_diff = paired["tabiclv2"] - paired["tabpfn3"]

    # Leave-one-dataset-out sign stability for the primary NLL effect.
    loo = {d: float(nll.drop(index=d).mean()) for d in nll.index}

    # Verify checksum entries that are present in the final manifest.
    checksum_file = root / "sha256sums_final.json"
    checksum_report = {"checked": 0, "mismatches": {}, "missing_files": []}
    if checksum_file.exists():
        stored = json.loads(checksum_file.read_text(encoding="utf-8"))
        for rel, expected in stored.items():
            path = root / rel
            if not path.exists():
                checksum_report["missing_files"].append(rel)
                continue
            actual = sha256(path)
            checksum_report["checked"] += 1
            if actual != expected:
                checksum_report["mismatches"][rel] = {"expected": expected, "actual": actual}

    admission = json.loads((root / "admission_preflight.json").read_text(encoding="utf-8"))
    rare = []
    for dataset, info in admission.get("datasets", {}).items():
        minimum = min(info.get("min_class_a", 10**9), info.get("min_class_b", 10**9))
        if minimum <= 5:
            rare.append({"dataset": dataset, "minimum_target_class_support": int(minimum)})

    report = {
        "evidence_root": str(root),
        "missing_required_files": missing,
        "row_count": int(len(df)),
        "expected_row_count": EXPECTED_ROWS,
        "row_count_ok": row_count_ok,
        "dataset_model_cells": int(cell_count),
        "fold_cells": int(fold_count),
        "methods_per_fold_min": int(methods_per_fold.min()),
        "methods_per_fold_max": int(methods_per_fold.max()),
        "primary_nll": {
            "mean_selective_minus_arithmetic": float(nll.mean()),
            "bootstrap_95_ci": [float(nll_ci[0]), float(nll_ci[1])],
            "selective_wins": int((nll < 0).sum()),
            "n_datasets": int(len(nll)),
            "wilcoxon_two_sided_p": nll_p,
            "leave_one_dataset_out_means": loo,
        },
        "metrics": metrics,
        "model_specific_nll_effect": {k: float(v) for k, v in model_means.items()},
        "tabiclv2_minus_tabpfn3_effect": {
            "mean": float(model_diff.mean()),
            "wilcoxon_two_sided_p": float(wilcoxon(model_diff.to_numpy(), alternative="two-sided").pvalue),
        },
        "rare_class_datasets_support_le_5": rare,
        "checksums": checksum_report,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    hard_fail = bool(missing or not row_count_ok or methods_per_fold.min() != EXPECTED_METHODS or methods_per_fold.max() != EXPECTED_METHODS or checksum_report["mismatches"])
    if hard_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
