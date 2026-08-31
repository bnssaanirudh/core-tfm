from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from core_tfm.data.openml import load_pair_dataset

from .common import json_dump, load_config, resolve_paths

DATASETS = ["anneal", "car", "credit", "customer", "diamonds", "marketing", "mic", "nursery", "phishing", "wine"]


def support_table():
    rows = []
    for name in DATASETS:
        ds = load_pair_dataset(name)
        ca = ds.a.astype(str).value_counts()
        cb = ds.b.astype(str).value_counts()
        rows.append({
            "dataset": name,
            "n_rows": int(len(ds.X)),
            "a_classes": int(ca.size),
            "b_classes": int(cb.size),
            "min_a_support": int(ca.min()),
            "min_b_support": int(cb.min()),
            "min_target_support": int(min(ca.min(), cb.min())),
            "a_supports": json.dumps({str(k): int(v) for k, v in ca.to_dict().items()}, sort_keys=True),
            "b_supports": json.dumps({str(k): int(v) for k, v in cb.to_dict().items()}, sort_keys=True),
        })
    return pd.DataFrame(rows)


def threshold_effects(all_folds: Path, support: pd.DataFrame, thresholds: list[int]):
    df = pd.read_csv(all_folds)
    primary = df[df["model"].isin(["tabiclv2", "tabpfn3"]) & (df["train_limit"] == 256)].copy()
    methods = set(primary["method"].astype(str))
    sel = "selective_core" if "selective_core" in methods else "selective"
    ari = "arithmetic"
    cell = (
        primary[primary["method"].isin([sel, ari])]
        .groupby(["seed", "dataset", "model", "method"], as_index=False)["joint_nll"].mean()
        .pivot(index=["seed", "dataset", "model"], columns="method", values="joint_nll")
        .reset_index()
    )
    cell["diff"] = cell[sel] - cell[ari]
    ds_seed = cell.groupby(["seed", "dataset"], as_index=False)["diff"].mean()
    support_map = support.set_index("dataset")["min_target_support"].to_dict()
    ds_seed["min_target_support"] = ds_seed["dataset"].map(support_map)

    rows = []
    for thr in thresholds:
        kept = ds_seed[ds_seed["min_target_support"] >= thr]
        ds_mean = kept.groupby("dataset", as_index=False)["diff"].mean()
        diff = ds_mean["diff"].to_numpy()
        p = None
        if len(diff) >= 2 and not np.allclose(diff, 0):
            try:
                p = float(wilcoxon(diff).pvalue)
            except Exception:
                pass
        rows.append({
            "minimum_support_threshold": int(thr),
            "datasets_kept": int(ds_mean["dataset"].nunique()),
            "dataset_names": ",".join(sorted(ds_mean["dataset"].unique())),
            "mean_selective_minus_arithmetic": float(diff.mean()) if len(diff) else np.nan,
            "dataset_wins": int((diff < 0).sum()) if len(diff) else 0,
            "dataset_losses": int((diff > 0).sum()) if len(diff) else 0,
            "wilcoxon_p": p,
        })
    return pd.DataFrame(rows), ds_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    paths = resolve_paths(args.repo, args.config)
    cfg = load_config(paths.config_path)
    out = paths.output_root / "analysis" / "rare_class"
    out.mkdir(parents=True, exist_ok=True)

    support = support_table()
    support.to_csv(out / "dataset_support.csv", index=False)

    all_folds = paths.output_root / "analysis" / "all_fold_results.csv"
    if not all_folds.exists():
        raise FileNotFoundError("Run aggregate.py before rare_class.py")
    effects, cells = threshold_effects(all_folds, support, [int(x) for x in cfg["rare_class_thresholds"]])
    effects.to_csv(out / "threshold_effects.csv", index=False)
    cells.to_csv(out / "dataset_seed_effects_with_support.csv", index=False)
    json_dump(out / "SUMMARY.json", {
        "thresholds": [int(x) for x in cfg["rare_class_thresholds"]],
        "datasets": int(len(support)),
        "lowest_support": int(support["min_target_support"].min()),
        "note": "Exclusion sensitivity uses the dataset-level minimum target-class support from the pinned pair datasets; model inference is not repeated for this post-hoc sensitivity analysis.",
    })
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
