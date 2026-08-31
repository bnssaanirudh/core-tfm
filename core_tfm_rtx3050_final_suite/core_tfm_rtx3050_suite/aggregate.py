from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .common import json_dump, load_config, resolve_paths

PRIMARY = ("tabiclv2", "tabpfn3")
SELECTIVE_NAMES = ("selective_core", "selective")
ARITHMETIC_NAMES = ("arithmetic",)
J1_NAMES = ("j1_b_then_a", "j1")
J2_NAMES = ("j2_a_then_b", "j2")


def resolve_method(df: pd.DataFrame, names):
    have = set(df["method"].astype(str).unique())
    for n in names:
        if n in have:
            return n
    raise ValueError(f"None of {names} found. Available methods: {sorted(have)}")


def discover_runs(root: Path):
    rows = []
    for csv in root.rglob("fold_results.csv"):
        rel = csv.parent.relative_to(root)
        seed = train = None
        group = rel.parts[0] if rel.parts else "unknown"
        leaf = rel.parts[-1] if rel.parts else ""
        if leaf.startswith("seed_") and "_train_" in leaf:
            a, b = leaf.split("_train_", 1)
            try:
                seed = int(a.replace("seed_", ""))
                train = int(b)
            except ValueError:
                pass
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if df.empty:
            continue
        df["suite_group"] = group
        df["seed"] = seed
        df["train_limit"] = train
        df["run_dir"] = str(csv.parent)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paired_effects(df: pd.DataFrame, train_limit: int | None = None):
    x = df[df["model"].isin(PRIMARY)].copy()
    if train_limit is not None:
        x = x[x["train_limit"] == train_limit]
    sel = resolve_method(x, SELECTIVE_NAMES)
    ari = resolve_method(x, ARITHMETIC_NAMES)
    piv = (
        x[x["method"].isin([sel, ari])]
        .groupby(["seed", "train_limit", "dataset", "model", "method"], as_index=False)["joint_nll"].mean()
        .pivot(index=["seed", "train_limit", "dataset", "model"], columns="method", values="joint_nll")
        .reset_index()
    )
    piv["selective_minus_arithmetic"] = piv[sel] - piv[ari]
    return piv


def view_reliability(df: pd.DataFrame):
    x = df[df["model"].isin(PRIMARY)].copy()
    j1 = resolve_method(x, J1_NAMES)
    j2 = resolve_method(x, J2_NAMES)
    keys = ["seed", "train_limit", "dataset", "model", "fold"]
    a = x[x["method"] == j1][keys + ["joint_nll", "conditional_nll_a_given_b", "factorization_tv"]].rename(
        columns={"joint_nll": "j1_joint_nll", "conditional_nll_a_given_b": "a_given_b_nll", "factorization_tv": "factorization_tv_j1"}
    )
    b = x[x["method"] == j2][keys + ["joint_nll", "conditional_nll_b_given_a", "factorization_tv"]].rename(
        columns={"joint_nll": "j2_joint_nll", "conditional_nll_b_given_a": "b_given_a_nll", "factorization_tv": "factorization_tv_j2"}
    )
    out = a.merge(b, on=keys, how="inner")
    # Exact factorization identities: NLL(J1)=NLL(p_B)+NLL(p_A|B), and similarly for J2.
    out["direct_b_nll"] = out["j1_joint_nll"] - out["a_given_b_nll"]
    out["direct_a_nll"] = out["j2_joint_nll"] - out["b_given_a_nll"]
    out["factorization_tv"] = out[["factorization_tv_j1", "factorization_tv_j2"]].mean(axis=1)

    eff = paired_effects(df)
    eff_cell = eff.groupby(["seed", "train_limit", "dataset", "model"], as_index=False)["selective_minus_arithmetic"].mean()
    out = out.merge(eff_cell, on=["seed", "train_limit", "dataset", "model"], how="left")
    return out


def summarize_multi_seed(effects: pd.DataFrame, train_limit: int):
    x = effects[effects["train_limit"] == train_limit].copy()
    if x.empty:
        return pd.DataFrame(), {}
    dataset_seed = x.groupby(["seed", "dataset"], as_index=False)["selective_minus_arithmetic"].mean()
    dataset_mean = dataset_seed.groupby("dataset", as_index=False)["selective_minus_arithmetic"].mean()
    diff = dataset_mean["selective_minus_arithmetic"].to_numpy()
    p = None
    if len(diff) >= 2 and not np.allclose(diff, 0):
        try:
            p = float(wilcoxon(diff).pvalue)
        except Exception:
            p = None
    summary = {
        "train_limit": train_limit,
        "n_seeds": int(dataset_seed["seed"].nunique()),
        "n_datasets": int(dataset_seed["dataset"].nunique()),
        "mean_selective_minus_arithmetic": float(diff.mean()),
        "dataset_wins": int((diff < 0).sum()),
        "dataset_losses": int((diff > 0).sum()),
        "wilcoxon_p_dataset_averaged_across_seeds": p,
    }
    return dataset_seed, summary


def aggregate_auxiliary(root: Path):
    abl, sens = [], []
    for p in root.rglob("selection_ablations.csv"):
        try:
            d = pd.read_csv(p)
            if not d.empty:
                leaf = p.parent.name
                seed, train = None, None
                if leaf.startswith("seed_") and "_train_" in leaf:
                    s, t = leaf.split("_train_", 1); seed = int(s.replace("seed_", "")); train = int(t)
                d["seed"] = seed; d["train_limit"] = train; d["run_dir"] = str(p.parent)
                abl.append(d)
        except Exception:
            pass
    for p in root.rglob("validation_fraction_sensitivity.csv"):
        try:
            d = pd.read_csv(p)
            if not d.empty:
                leaf = p.parent.name
                seed, train = None, None
                if leaf.startswith("seed_") and "_train_" in leaf:
                    s, t = leaf.split("_train_", 1); seed = int(s.replace("seed_", "")); train = int(t)
                d["seed"] = seed; d["train_limit"] = train; d["run_dir"] = str(p.parent)
                sens.append(d)
        except Exception:
            pass
    return (pd.concat(abl, ignore_index=True) if abl else pd.DataFrame(), pd.concat(sens, ignore_index=True) if sens else pd.DataFrame())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    paths = resolve_paths(args.repo, args.config)
    cfg = load_config(paths.config_path)
    out = paths.output_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    folds = discover_runs(paths.output_root)
    if folds.empty:
        raise RuntimeError("No fold_results.csv files found under the suite output root.")
    folds.to_csv(out / "all_fold_results.csv", index=False)

    effects = paired_effects(folds)
    effects.to_csv(out / "selective_vs_arithmetic_by_model.csv", index=False)

    multi_seed_rows, multi_seed_summary = summarize_multi_seed(effects, int(cfg["multi_seed"]["train_limit"]))
    multi_seed_rows.to_csv(out / "multi_seed_dataset_effects.csv", index=False)
    json_dump(out / "multi_seed_summary.json", multi_seed_summary)

    context = effects.groupby(["train_limit", "seed"], as_index=False)["selective_minus_arithmetic"].mean()
    context.to_csv(out / "context_seed_effects.csv", index=False)
    context_summary = effects.groupby("train_limit", as_index=False).agg(
        mean_selective_minus_arithmetic=("selective_minus_arithmetic", "mean"),
        std=("selective_minus_arithmetic", "std"),
        n_cells=("selective_minus_arithmetic", "size"),
    )
    context_summary.to_csv(out / "context_summary.csv", index=False)

    model = effects.groupby(["model", "train_limit"], as_index=False).agg(
        mean_selective_minus_arithmetic=("selective_minus_arithmetic", "mean"),
        n=("selective_minus_arithmetic", "size"),
    )
    model.to_csv(out / "model_effects.csv", index=False)

    vr = view_reliability(folds)
    vr.to_csv(out / "view_reliability.csv", index=False)
    vr_summary = vr.groupby(["model", "train_limit"], as_index=False)[
        ["direct_a_nll", "direct_b_nll", "a_given_b_nll", "b_given_a_nll", "factorization_tv", "selective_minus_arithmetic"]
    ].mean()
    vr_summary.to_csv(out / "view_reliability_summary.csv", index=False)

    correlations = {}
    for model_name, g in vr.groupby("model"):
        numeric = g[["direct_a_nll", "direct_b_nll", "a_given_b_nll", "b_given_a_nll", "factorization_tv", "selective_minus_arithmetic"]]
        correlations[model_name] = numeric.corr(method="spearman")["selective_minus_arithmetic"].drop("selective_minus_arithmetic").to_dict()
    json_dump(out / "view_reliability_correlations.json", correlations)

    abl, sens = aggregate_auxiliary(paths.output_root)
    if not abl.empty:
        abl.to_csv(out / "all_selection_ablations.csv", index=False)
        if "family" in abl.columns and "joint_nll" in abl.columns:
            abl.groupby(["train_limit", "family"], as_index=False)["joint_nll"].mean().to_csv(out / "selection_ablation_summary.csv", index=False)
    if not sens.empty:
        sens.to_csv(out / "all_validation_fraction_sensitivity.csv", index=False)
        metric = "joint_nll" if "joint_nll" in sens.columns else None
        frac = "validation_fraction" if "validation_fraction" in sens.columns else ("fraction" if "fraction" in sens.columns else None)
        if metric and frac:
            sens.groupby(["train_limit", frac], as_index=False)[metric].mean().to_csv(out / "validation_fraction_summary.csv", index=False)

    summary = {
        "fold_rows": int(len(folds)),
        "runs_found": int(folds["run_dir"].nunique()),
        "multi_seed": multi_seed_summary,
        "context_train_limits_found": sorted(int(x) for x in folds["train_limit"].dropna().unique()),
        "selection_ablation_rows": int(len(abl)),
        "validation_sensitivity_rows": int(len(sens)),
        "view_reliability_rows": int(len(vr)),
    }
    json_dump(out / "ANALYSIS_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
