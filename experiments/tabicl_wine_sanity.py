"""One-fold TabICLv2 sanity benchmark on the source-paper Wine reconstruction.

Uses the exact pinned OpenML tables from Klötergens et al. (2026): red ID 40691
and white ID 40498, concatenated with source-table color as B. The white target
keeps its OpenML index encoding, so the merged A target has eight levels, matching
Appendix B of the source paper. This is a one-fold connectivity/protocol sanity
run before a five-fold benchmark.
"""
from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from core_tfm.data.openml import load_pair_dataset
from core_tfm.inference.extract import extract_pair_predictions
from core_tfm.metrics.distributions import marginal_distortion, total_variation
from core_tfm.metrics.scoring import joint_brier, joint_log_loss
from core_tfm.models.tfm_adapters import tabiclv2_adapter
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool, independent_joint
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.soft import soft_reconciliation

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def factory():
    return tabiclv2_adapter(
        device="cpu",
        n_estimators=1,
        kv_cache=True,
        checkpoint_version=CHECKPOINT,
        random_state=42,
        n_jobs=4,
        verbose=False,
    )


def main():
    data = load_pair_dataset("wine")
    X, a, b = data.X, data.a, data.b
    tr, te = next(
        iter(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, a))
    )

    out = extract_pair_predictions(
        factory,
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
    rows = []
    for name, q in methods.items():
        rows.append(
            {
                "method": name,
                "joint_nll": float(joint_log_loss(q, out.y_a_encoded, out.y_b_encoded)),
                "joint_brier": float(joint_brier(q, out.y_a_encoded, out.y_b_encoded)),
                "marginal_distortion": float(
                    marginal_distortion(q, p.p_a, p.p_b).mean()
                ),
            }
        )

    tv = total_variation(j1, j2)
    result = {
        "status": "sanity-only",
        "dataset": "Wine (OpenML 40691 + 40498)",
        "model": "TabICLv2",
        "model_config": {
            "checkpoint_version": CHECKPOINT,
            "device": "cpu",
            "n_estimators": 1,
            "kv_cache": True,
            "random_state": 42,
        },
        "environment": {
            "python": sys.version.split()[0],
            "tabicl": importlib.metadata.version("tabicl"),
            "torch": importlib.metadata.version("torch"),
        },
        "fold": 1,
        "n_total": int(len(X)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "k_a": int(len(out.classes_a)),
        "k_b": int(len(out.classes_b)),
        "source_paper_tabiclv2_factorization_tv": {"mean": 0.0193, "fold_std": 0.0007},
        "factorization_tv_mean": float(tv.mean()),
        "factorization_tv_median": float(np.median(tv)),
        "factorization_tv_max": float(tv.max()),
        "marginalization_tv_a_mean": float(
            total_variation(j1.sum(axis=2), p.p_a, axis=1).mean()
        ),
        "marginalization_tv_b_mean": float(
            total_variation(j2.sum(axis=1), p.p_b, axis=1).mean()
        ),
        "methods": rows,
    }
    path = Path("results/real_sanity_wine_tabiclv2.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
