"""Post-hoc reliability-aware analyses for the completed bounded benchmark.

This script never overwrites the archived Q1 benchmark. It reads fold_results.csv
and writes derived analyses to a new output directory. It also runs analytic
counterexamples that require no TFM inference.

The frozen benchmark uses canonical method names such as ``selective_core``,
``hard_core`` and ``soft_core_lambda_1``. Older development code sometimes used
short aliases (``selective``, ``mpr``, ``soft``). This module resolves aliases
explicitly and fails loudly instead of silently writing empty CSV files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core_tfm.research_extensions import inconsistency_accuracy_counterexamples

PRIMARY_MODELS = ("tabiclv2", "tabpfn3")
METHOD_ALIASES = {
    "selective": ("selective_core", "selective"),
    "arithmetic": ("arithmetic",),
    "geometric": ("geometric",),
    "j1": ("j1_b_then_a", "j1"),
    "j2": ("j2_a_then_b", "j2"),
    "hard": ("hard_core", "mpr", "hard"),
    "soft": ("soft_core_lambda_1", "soft", "soft_core_1"),
}


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_method(methods: set[str], key: str, *, required: bool = True) -> str | None:
    for candidate in METHOD_ALIASES[key]:
        if candidate in methods:
            return candidate
    if required:
        raise ValueError(
            f"Could not resolve method alias {key!r}. Available methods: {sorted(methods)}"
        )
    return None


def _resolved_methods(df: pd.DataFrame) -> dict[str, str | None]:
    methods = set(df["method"].astype(str).unique())
    return {
        key: _resolve_method(methods, key, required=key in {"selective", "arithmetic", "geometric", "j1", "j2"})
        for key in METHOD_ALIASES
    }


def _require_nonempty(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError(
            f"Derived table {name} is empty. Refusing to write an empty evidence CSV."
        )
    return frame


def oracle_decomposition(df: pd.DataFrame, names: dict[str, str | None]) -> pd.DataFrame:
    selected = str(names["selective"])
    arithmetic = str(names["arithmetic"])
    rows = []
    for (dataset, model, fold), g in df.groupby(["dataset", "model", "fold"], sort=True):
        losses = g.set_index("method")["joint_nll"].to_dict()
        if arithmetic not in losses or selected not in losses:
            continue
        oracle_method = min(losses, key=losses.get)
        oracle_loss = float(losses[oracle_method])
        rows.append({
            "dataset": dataset,
            "model": model,
            "fold": int(fold),
            "arithmetic_nll": float(losses[arithmetic]),
            "selective_nll": float(losses[selected]),
            "oracle_method": oracle_method,
            "oracle_nll": oracle_loss,
            "available_opportunity": float(losses[arithmetic] - oracle_loss),
            "selection_regret": float(losses[selected] - oracle_loss),
            "selective_minus_arithmetic": float(losses[selected] - losses[arithmetic]),
        })
    return _require_nonempty(pd.DataFrame(rows), "oracle_selection_decomposition")


def inconsistency_gain_analysis(df: pd.DataFrame, names: dict[str, str | None]) -> pd.DataFrame:
    selected = str(names["selective"])
    arithmetic = str(names["arithmetic"])
    base = df[df["method"] == arithmetic][["dataset", "model", "fold", "factorization_tv"]]
    sel = df[df["method"] == selected][["dataset", "model", "fold", "joint_nll"]].rename(
        columns={"joint_nll": "selective_nll"}
    )
    ari = df[df["method"] == arithmetic][["dataset", "model", "fold", "joint_nll"]].rename(
        columns={"joint_nll": "arithmetic_nll"}
    )
    out = base.merge(sel, on=["dataset", "model", "fold"]).merge(
        ari, on=["dataset", "model", "fold"]
    )
    out["selective_gain"] = out["arithmetic_nll"] - out["selective_nll"]
    return _require_nonempty(out, "inconsistency_vs_gain")


def model_reliability_proxy(df: pd.DataFrame, names: dict[str, str | None]) -> pd.DataFrame:
    j1_name = str(names["j1"])
    j2_name = str(names["j2"])
    raw = df[df["method"].isin([j1_name, j2_name])].copy()
    rows = []
    for (dataset, model, fold), g in raw.groupby(["dataset", "model", "fold"], sort=True):
        if set(g["method"]) != {j1_name, j2_name}:
            continue
        by = g.set_index("method")
        j1 = by.loc[j1_name]
        j2 = by.loc[j2_name]
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
    return _require_nonempty(pd.DataFrame(rows), "model_view_reliability_proxy")


def family_complexity_analysis(df: pd.DataFrame, names: dict[str, str | None]) -> pd.DataFrame:
    arithmetic = str(names["arithmetic"])
    geometric = str(names["geometric"])
    j1_name = str(names["j1"])
    j2_name = str(names["j2"])
    selected = str(names["selective"])
    hard = names["hard"]
    soft = names["soft"]

    family_defs = {
        "fallback": [arithmetic],
        "raw_pool": [j1_name, j2_name, arithmetic, geometric],
        "repair": [x for x in [arithmetic, geometric, hard, soft] if x],
        "full_archived": [x for x in [j1_name, j2_name, arithmetic, geometric, hard, soft, selected] if x],
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
    return _require_nonempty(pd.DataFrame(rows), "candidate_family_headroom")


def policy_transfer_analysis(df: pd.DataFrame) -> pd.DataFrame:
    primary = df[df["model"].isin(PRIMARY_MODELS)]
    cell = primary.groupby(["dataset", "model", "method"], as_index=False)["joint_nll"].mean()
    rows = []
    for model, mg in cell.groupby("model"):
        method_means = mg.groupby("method")["joint_nll"].mean().sort_values()
        for rank, (method, mean_nll) in enumerate(method_means.items(), 1):
            rows.append({
                "model": model,
                "rank": rank,
                "method": method,
                "mean_dataset_nll": float(mean_nll),
            })
    return _require_nonempty(pd.DataFrame(rows), "model_level_policy_transfer_proxy")


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
    parser.add_argument(
        "--fold-results",
        type=Path,
        default=Path("results/q1_fast_complete_256_v1/fold_results.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/reliability_aware_v1")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.fold_results)
    required = {"dataset", "model", "fold", "method", "joint_nll", "factorization_tv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    names = _resolved_methods(df)
    print("Resolved benchmark method names:", names)

    oracle = oracle_decomposition(df, names)
    oracle.to_csv(args.output / "oracle_selection_decomposition.csv", index=False)

    ig = inconsistency_gain_analysis(df, names)
    ig.to_csv(args.output / "inconsistency_vs_gain.csv", index=False)
    _write_json(
        args.output / "inconsistency_vs_gain_correlations.json",
        summarize_correlations(ig),
    )

    reliability = model_reliability_proxy(df, names)
    reliability.to_csv(args.output / "model_view_reliability_proxy.csv", index=False)

    complexity = family_complexity_analysis(df, names)
    complexity.to_csv(args.output / "candidate_family_headroom.csv", index=False)

    transfer = policy_transfer_analysis(df)
    transfer.to_csv(args.output / "model_level_policy_transfer_proxy.csv", index=False)

    _write_json(
        args.output / "analytic_tv_accuracy_counterexamples.json",
        inconsistency_accuracy_counterexamples(),
    )

    summary = {
        "source": str(args.fold_results),
        "rows": int(len(df)),
        "primary_models": list(PRIMARY_MODELS),
        "resolved_methods": names,
        "oracle_rows": int(len(oracle)),
        "inconsistency_gain_rows": int(len(ig)),
        "reliability_proxy_rows": int(len(reliability)),
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
