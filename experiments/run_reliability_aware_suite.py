"""Post-hoc reliability-aware analyses for the completed bounded benchmark.

This script never overwrites the archived Q1 benchmark. It reads fold_results.csv
and writes derived analyses to a new output directory. It also runs analytic
counterexamples that require no TFM inference.

Example
-------
python experiments/run_reliability_aware_suite.py \
  --fold-results results/q1_fast_complete_256_v1/fold_results.csv \
  --output results/reliability_aware_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from core_tfm.research_extensions import inconsistency_accuracy_counterexamples

PRIMARY_MODELS = ("tabiclv2", "tabpfn3")


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def oracle_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, fold), g in df.groupby(["dataset", "model", "fold"], sort=True):
        losses = g.set_index("method")["joint_nll"].to_dict()
        if "arithmetic" not in losses or "selective" not in losses:
            continue
        oracle_method = min(losses, key=losses.get)
        oracle_loss = float(losses[oracle_method])
        rows.append({
            "dataset": dataset,
            "model": model,
            "fold": int(fold),
            "arithmetic_nll": float(losses["arithmetic"]),
            "selective_nll": float(losses["selective"]),
            "oracle_method": oracle_method,
            "oracle_nll": oracle_loss,
            "available_opportunity": float(losses["arithmetic"] - oracle_loss),
            "selection_regret": float(losses["selective"] - oracle_loss),
            "selective_minus_arithmetic": float(losses["selective"] - losses["arithmetic"]),
        })
    return pd.DataFrame(rows)


def inconsistency_gain_analysis(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["method"] == "arithmetic"][["dataset", "model", "fold", "factorization_tv"]]
    sel = df[df["method"] == "selective"][["dataset", "model", "fold", "joint_nll"]].rename(columns={"joint_nll": "selective_nll"})
    ari = df[df["method"] == "arithmetic"][["dataset", "model", "fold", "joint_nll"]].rename(columns={"joint_nll": "arithmetic_nll"})
    out = base.merge(sel, on=["dataset", "model", "fold"]).merge(ari, on=["dataset", "model", "fold"])
    out["selective_gain"] = out["arithmetic_nll"] - out["selective_nll"]
    return out


def model_reliability_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Use existing raw-joint conditional losses to diagnose model heterogeneity.

    This is deliberately a proxy analysis because the archived table does not
    contain raw direct-marginal NLLs. New seeded runs should archive those view-
    level metrics explicitly; see configs/reliability_aware_experiments.yaml.
    """
    raw = df[df["method"].isin(["j1_b_then_a", "j2_a_then_b"])].copy()
    rows = []
    for (dataset, model, fold), g in raw.groupby(["dataset", "model", "fold"], sort=True):
        if set(g["method"]) != {"j1_b_then_a", "j2_a_then_b"}:
            continue
        by = g.set_index("method")
        j1 = by.loc["j1_b_then_a"]
        j2 = by.loc["j2_a_then_b"]
        rows.append({
            "dataset": dataset,
            "model": model,
            "fold": int(fold),
            "j1_joint_nll": float(j1.joint_nll),
            "j2_joint_nll": float(j2.joint_nll),
            "j1_conditional_nll_a_given_b": float(j1.conditional_nll_a_given_b),
            "j2_conditional_nll_b_given_a": float(j2.conditional_nll_b_given_a),
            "raw_direction_gap": float(j1.joint_nll - j2.joint_nll),
            "factorization_tv": float(j1.factorization_tv),
        })
    return pd.DataFrame(rows)


def family_complexity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize archived method families as evidence for safe selection.

    Exact validation-time SRM selection requires candidate-level validation
    scores, which are produced by the new rerun scaffold. Here we quantify test
    headroom of progressively richer families using the existing archive.
    """
    family_defs = {
        "fallback": ["arithmetic"],
        "raw_pool": ["j1_b_then_a", "j2_a_then_b", "arithmetic", "geometric"],
        "repair": ["arithmetic", "geometric", "mpr", "soft"],
        "full_archived": ["j1_b_then_a", "j2_a_then_b", "arithmetic", "geometric", "mpr", "soft", "selective"],
    }
    rows = []
    for (dataset, model, fold), g in df.groupby(["dataset", "model", "fold"], sort=True):
        losses = g.set_index("method")["joint_nll"].to_dict()
        for family, methods in family_defs.items():
            available = {m: losses[m] for m in methods if m in losses}
            if not available:
                continue
            best_method = min(available, key=available.get)
            rows.append({
                "dataset": dataset,
                "model": model,
                "fold": int(fold),
                "family": family,
                "family_size_observed": len(available),
                "oracle_method_within_family": best_method,
                "oracle_nll_within_family": float(available[best_method]),
            })
    return pd.DataFrame(rows)


def policy_transfer_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate transfer of one globally chosen method across datasets.

    This is a method-level transfer diagnostic using archived test scores, not a
    deployable validation-transfer estimate. A proper leave-one-dataset-out
    policy-transfer experiment is scaffolded for the new reruns.
    """
    primary = df[df["model"].isin(PRIMARY_MODELS)]
    cell = primary.groupby(["dataset", "model", "method"], as_index=False)["joint_nll"].mean()
    rows = []
    for model, mg in cell.groupby("model"):
        method_means = mg.groupby("method")["joint_nll"].mean().sort_values()
        for rank, (method, mean_nll) in enumerate(method_means.items(), 1):
            rows.append({"model": model, "rank": rank, "method": method, "mean_dataset_nll": float(mean_nll)})
    return pd.DataFrame(rows)


def summarize_correlations(ig: pd.DataFrame) -> dict:
    out = {}
    for model, g in ig.groupby("model"):
        if len(g) < 3:
            continue
        out[model] = {
            "pearson_tv_vs_gain": float(g["factorization_tv"].corr(g["selective_gain"], method="pearson")),
            "spearman_tv_vs_gain": float(g["factorization_tv"].corr(g["selective_gain"], method="spearman")),
            "n": int(len(g)),
        }
    if len(ig) >= 3:
        out["all"] = {
            "pearson_tv_vs_gain": float(ig["factorization_tv"].corr(ig["selective_gain"], method="pearson")),
            "spearman_tv_vs_gain": float(ig["factorization_tv"].corr(ig["selective_gain"], method="spearman")),
            "n": int(len(ig)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-results", type=Path, default=Path("results/q1_fast_complete_256_v1/fold_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/reliability_aware_v1"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.fold_results)
    required = {"dataset", "model", "fold", "method", "joint_nll", "factorization_tv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    oracle = oracle_decomposition(df)
    oracle.to_csv(args.output / "oracle_selection_decomposition.csv", index=False)

    ig = inconsistency_gain_analysis(df)
    ig.to_csv(args.output / "inconsistency_vs_gain.csv", index=False)
    _write_json(args.output / "inconsistency_vs_gain_correlations.json", summarize_correlations(ig))

    reliability = model_reliability_proxy(df)
    reliability.to_csv(args.output / "model_view_reliability_proxy.csv", index=False)

    complexity = family_complexity_analysis(df)
    complexity.to_csv(args.output / "candidate_family_headroom.csv", index=False)

    transfer = policy_transfer_analysis(df)
    transfer.to_csv(args.output / "model_level_policy_transfer_proxy.csv", index=False)

    _write_json(args.output / "analytic_tv_accuracy_counterexamples.json", inconsistency_accuracy_counterexamples())

    summary = {
        "source": str(args.fold_results),
        "rows": int(len(df)),
        "primary_models": list(PRIMARY_MODELS),
        "note": "Derived analyses only; no new TFM inference was performed by this script.",
        "next_runs": [
            "multi-seed reruns",
            "context-size sensitivity",
            "third-TFM benchmark",
            "direct marginal/conditional view calibration archive",
            "inference-perturbation dispersion archive",
            "validation-score traces for Safe Selective CoRe",
            "rare-class sensitivity and support-adaptive penalties",
            "known-truth downstream utility experiment",
        ],
    }
    _write_json(args.output / "RUN_METADATA.json", summary)
    print(f"Wrote reliability-aware analyses to {args.output}")


if __name__ == "__main__":
    main()
