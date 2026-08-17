"""Five-fold real-model CoRe-TFM benchmark on UCI Car Evaluation using TabICLv2.

Targets follow the source consistency protocol: A=class, B=safety.  The remaining
five categorical columns are X. Splits use deterministic StratifiedKFold on A
with shuffle=True, random_state=42. This is a small real-model benchmark rather
than an exact reproduction of the source paper because its fold seed is not
published and this run uses n_estimators=1 for CPU tractability.

UCI dataset DOI: 10.24432/C5JP48 (CC BY 4.0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from ucimlrepo import fetch_ucirepo

from core_tfm.inference.extract import extract_pair_predictions
from core_tfm.metrics.distributions import marginal_distortion, total_variation
from core_tfm.metrics.scoring import conditional_log_losses, joint_brier, joint_log_loss
from core_tfm.models.tfm_adapters import tabiclv2_adapter
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool, independent_joint
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.soft import soft_reconciliation


def _factory():
    return tabiclv2_adapter(
        device="cpu",
        n_estimators=1,
        kv_cache=True,
        random_state=42,
        n_jobs=4,
        verbose=False,
    )


def _evaluate_fold(X, a, b, tr, te, fold):
    out = extract_pair_predictions(
        _factory,
        X.iloc[tr].reset_index(drop=True),
        a.iloc[tr].reset_index(drop=True),
        b.iloc[tr].reset_index(drop=True),
        X.iloc[te].reset_index(drop=True),
        a_test=a.iloc[te].reset_index(drop=True),
        b_test=b.iloc[te].reset_index(drop=True),
    )
    p = out.predictions
    j1, j2 = p.j_b_then_a, p.j_a_then_b
    methods = {
        "j1_b_then_a": j1,
        "j2_a_then_b": j2,
        "independent": independent_joint(p.p_a, p.p_b),
        "arithmetic": arithmetic_pool(j1, j2),
        "geometric": geometric_pool(j1, j2),
        "hard_core": marginal_preserving_reconciliation(j1, j2, p.p_a, p.p_b).joint,
        "soft_core_lambda_1": soft_reconciliation(
            j1, j2, p.p_a, p.p_b, lambda_a=1, lambda_b=1
        ).joint,
    }

    method_rows = []
    for name, q in methods.items():
        ca, cb = conditional_log_losses(q, out.y_a_encoded, out.y_b_encoded)
        method_rows.append(
            {
                "method": name,
                "joint_nll": float(joint_log_loss(q, out.y_a_encoded, out.y_b_encoded)),
                "joint_brier": float(joint_brier(q, out.y_a_encoded, out.y_b_encoded)),
                "a_given_b_nll": float(ca),
                "b_given_a_nll": float(cb),
                "marginal_distortion": float(
                    marginal_distortion(q, p.p_a, p.p_b).mean()
                ),
            }
        )

    tv = total_variation(j1, j2)
    return {
        "fold": int(fold),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "k_a": int(len(out.classes_a)),
        "k_b": int(len(out.classes_b)),
        "factorization_tv_mean": float(tv.mean()),
        "factorization_tv_median": float(np.median(tv)),
        "factorization_tv_max": float(tv.max()),
        "marginalization_tv_a_mean": float(
            total_variation(j1.sum(axis=2), p.p_a, axis=1).mean()
        ),
        "marginalization_tv_b_mean": float(
            total_variation(j2.sum(axis=1), p.p_b, axis=1).mean()
        ),
        "methods": method_rows,
    }


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def main():
    ds = fetch_ucirepo(id=19)
    Xall = ds.data.features.copy()
    a = ds.data.targets.iloc[:, 0].astype("category")
    b = Xall["safety"].astype("category")
    X = Xall.drop(columns=["safety"])
    for col in X.columns:
        X[col] = X[col].astype("category")

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = [
        _evaluate_fold(X, a, b, tr, te, fold=i)
        for i, (tr, te) in enumerate(splitter.split(X, a), start=1)
    ]

    metric_names = [
        "factorization_tv_mean",
        "marginalization_tv_a_mean",
        "marginalization_tv_b_mean",
    ]
    aggregate = {
        metric: _mean_std([fold[metric] for fold in folds]) for metric in metric_names
    }

    method_names = [row["method"] for row in folds[0]["methods"]]
    method_metrics = [
        "joint_nll",
        "joint_brier",
        "a_given_b_nll",
        "b_given_a_nll",
        "marginal_distortion",
    ]
    aggregate["methods"] = []
    for method in method_names:
        row = {"method": method}
        for metric in method_metrics:
            vals = [
                next(r for r in fold["methods"] if r["method"] == method)[metric]
                for fold in folds
            ]
            row[metric] = _mean_std(vals)
        aggregate["methods"].append(row)

    nll_ranking = sorted(
        (
            (row["joint_nll"]["mean"], row["method"])
            for row in aggregate["methods"]
        )
    )

    result = {
        "status": "small-real-benchmark",
        "dataset": "UCI Car Evaluation",
        "dataset_doi": "10.24432/C5JP48",
        "model": "TabICLv2",
        "model_config": {
            "device": "cpu",
            "n_estimators": 1,
            "kv_cache": True,
            "random_state": 42,
        },
        "split_config": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "stratify_target": "class",
        },
        "source_protocol_note": (
            "Targets match the source paper, but this is not an exact reproduction: "
            "the source fold seed is unpublished and this CPU run uses n_estimators=1."
        ),
        "source_paper_tabiclv2_factorization_tv": {
            "mean": 0.0644,
            "fold_std": 0.0058,
        },
        "folds": folds,
        "aggregate": aggregate,
        "best_mean_joint_nll_method": nll_ranking[0][1],
        "best_mean_joint_nll": float(nll_ranking[0][0]),
    }

    path = Path("results/real_car_tabiclv2_fivefold.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
