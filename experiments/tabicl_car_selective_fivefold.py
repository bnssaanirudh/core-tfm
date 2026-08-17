"""Leakage-free five-fold Selective CoRe benchmark on TabICLv2 + UCI Car.

For each outer fold, the outer-training set is split into inner training and
validation data. The full Selective CoRe candidate family is chosen using only
inner-validation joint log loss. The four TabICLv2 probability views are then
refit on the complete outer-training fold and the frozen policy is evaluated on
the untouched outer-test fold.

This is a small released-model benchmark, not an exact reproduction of the
source consistency paper. UCI Car Evaluation DOI: 10.24432/C5JP48.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from ucimlrepo import fetch_ucirepo

from core_tfm.inference.extract import extract_pair_predictions
from core_tfm.metrics.distributions import marginal_distortion, total_variation
from core_tfm.metrics.scoring import joint_brier, joint_log_loss
from core_tfm.models.tfm_adapters import tabiclv2_adapter
from core_tfm.reconciliation.selective import (
    apply_reconciliation_policy,
    select_reconciliation_policy,
)

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PENALTIES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def _factory():
    return tabiclv2_adapter(
        device="cpu",
        n_estimators=1,
        kv_cache=True,
        checkpoint_version=CHECKPOINT,
        random_state=42,
        n_jobs=4,
        verbose=False,
    )


def _extract(X_train, a_train, b_train, X_eval, a_eval, b_eval):
    return extract_pair_predictions(
        _factory,
        X_train.reset_index(drop=True),
        a_train.reset_index(drop=True),
        b_train.reset_index(drop=True),
        X_eval.reset_index(drop=True),
        a_test=a_eval.reset_index(drop=True),
        b_test=b_eval.reset_index(drop=True),
    )


def _policy_dict(policy):
    return {
        "name": policy.name,
        "weight": None if policy.weight is None else float(policy.weight),
        "marginal_penalty": (
            None if policy.marginal_penalty is None else float(policy.marginal_penalty)
        ),
        "validation_joint_nll": float(policy.score),
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

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []

    for fold, (outer_train_idx, test_idx) in enumerate(outer.split(X, a), start=1):
        inner_train_idx, val_idx = train_test_split(
            outer_train_idx,
            test_size=0.20,
            random_state=1000 + fold,
            stratify=a.iloc[outer_train_idx],
        )

        val_out = _extract(
            X.iloc[inner_train_idx],
            a.iloc[inner_train_idx],
            b.iloc[inner_train_idx],
            X.iloc[val_idx],
            a.iloc[val_idx],
            b.iloc[val_idx],
        )
        vp = val_out.predictions
        selection = select_reconciliation_policy(
            vp.j_b_then_a,
            vp.j_a_then_b,
            vp.p_a,
            vp.p_b,
            val_out.y_a_encoded,
            val_out.y_b_encoded,
            weights=WEIGHTS,
            marginal_penalties=PENALTIES,
        )

        test_out = _extract(
            X.iloc[outer_train_idx],
            a.iloc[outer_train_idx],
            b.iloc[outer_train_idx],
            X.iloc[test_idx],
            a.iloc[test_idx],
            b.iloc[test_idx],
        )
        tp = test_out.predictions
        selected_joint = apply_reconciliation_policy(
            selection,
            tp.j_b_then_a,
            tp.j_a_then_b,
            tp.p_a,
            tp.p_b,
        )

        j1_nll = float(
            joint_log_loss(tp.j_b_then_a, test_out.y_a_encoded, test_out.y_b_encoded)
        )
        j2_nll = float(
            joint_log_loss(tp.j_a_then_b, test_out.y_a_encoded, test_out.y_b_encoded)
        )
        selected_nll = float(
            joint_log_loss(selected_joint, test_out.y_a_encoded, test_out.y_b_encoded)
        )
        selected_brier = float(
            joint_brier(selected_joint, test_out.y_a_encoded, test_out.y_b_encoded)
        )
        tv = total_variation(tp.j_b_then_a, tp.j_a_then_b)

        folds.append(
            {
                "fold": fold,
                "n_outer_train": int(len(outer_train_idx)),
                "n_inner_train": int(len(inner_train_idx)),
                "n_validation": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "selected_policy": _policy_dict(selection.policy),
                "factorization_tv_mean": float(tv.mean()),
                "test_joint_nll_selected": selected_nll,
                "test_joint_brier_selected": selected_brier,
                "test_marginal_distortion_selected": float(
                    marginal_distortion(selected_joint, tp.p_a, tp.p_b).mean()
                ),
                "test_joint_nll_j1": j1_nll,
                "test_joint_nll_j2": j2_nll,
                "test_joint_nll_best_original": min(j1_nll, j2_nll),
                "selected_minus_best_original_nll": selected_nll - min(j1_nll, j2_nll),
            }
        )

    policy_counts = Counter(fold["selected_policy"]["name"] for fold in folds)
    aggregate = {
        "factorization_tv_mean": _mean_std(
            [fold["factorization_tv_mean"] for fold in folds]
        ),
        "selected_joint_nll": _mean_std(
            [fold["test_joint_nll_selected"] for fold in folds]
        ),
        "selected_joint_brier": _mean_std(
            [fold["test_joint_brier_selected"] for fold in folds]
        ),
        "selected_marginal_distortion": _mean_std(
            [fold["test_marginal_distortion_selected"] for fold in folds]
        ),
        "best_original_joint_nll": _mean_std(
            [fold["test_joint_nll_best_original"] for fold in folds]
        ),
        "selected_minus_best_original_nll": _mean_std(
            [fold["selected_minus_best_original_nll"] for fold in folds]
        ),
        "policy_name_counts": dict(sorted(policy_counts.items())),
        "selected_policy_is_original_folds": int(
            sum(
                fold["selected_policy"]["name"]
                in {"j1_b_then_a", "j2_a_then_b"}
                for fold in folds
            )
        ),
    }

    result = {
        "status": "small-real-selective-benchmark",
        "dataset": "UCI Car Evaluation",
        "dataset_doi": "10.24432/C5JP48",
        "model": "TabICLv2",
        "model_config": {
            "checkpoint_version": CHECKPOINT,
            "device": "cpu",
            "n_estimators": 1,
            "kv_cache": True,
            "random_state": 42,
            "n_jobs": 4,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tabicl": importlib.metadata.version("tabicl"),
            "torch": importlib.metadata.version("torch"),
        },
        "outer_split": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "stratify_target": "class",
        },
        "selection": {
            "validation_fraction": 0.20,
            "validation_seed_rule": "1000 + outer_fold",
            "weights": list(WEIGHTS),
            "marginal_penalties": list(PENALTIES),
            "candidate_family_note": (
                "Two original orders, arithmetic pool, weighted geometric pool, "
                "weighted hard CoRe, and weighted soft CoRe. Selection uses only "
                "inner-validation labels; outer-test labels are never used for selection."
            ),
        },
        "folds": folds,
        "aggregate": aggregate,
    }

    path = Path("results/real_car_tabiclv2_selective_fivefold.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
