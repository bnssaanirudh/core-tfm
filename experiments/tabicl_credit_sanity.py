"""One-fold real-model sanity check: Credit + TabICLv2.

This is intentionally a pipeline-validation experiment, not the final paper
benchmark. It uses two ensemble members on CPU to keep CI practical. Final
reported experiments should freeze the model configuration and run all five
folds (plus other TFMs/datasets).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

from core_tfm.data.openml import load_pair_dataset
from core_tfm.inference.extract import extract_pair_predictions
from core_tfm.metrics.distributions import total_variation, marginal_distortion
from core_tfm.metrics.scoring import joint_log_loss, joint_brier, conditional_log_losses
from core_tfm.models.tfm_adapters import tabiclv2_adapter
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool, independent_joint
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.soft import soft_reconciliation


def main():
    data = load_pair_dataset("credit")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr, te = next(iter(splitter.split(data.X, data.a)))

    def factory():
        return tabiclv2_adapter(
            device="cpu",
            n_estimators=2,
            kv_cache=True,
            random_state=42,
            n_jobs=4,
            verbose=False,
        )

    out = extract_pair_predictions(
        factory,
        data.X.iloc[tr].reset_index(drop=True),
        data.a.iloc[tr].reset_index(drop=True),
        data.b.iloc[tr].reset_index(drop=True),
        data.X.iloc[te].reset_index(drop=True),
        a_test=data.a.iloc[te].reset_index(drop=True),
        b_test=data.b.iloc[te].reset_index(drop=True),
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
            j1, j2, p.p_a, p.p_b, lambda_a=1.0, lambda_b=1.0
        ).joint,
    }

    rows = []
    for name, q in methods.items():
        ca, cb = conditional_log_losses(q, out.y_a_encoded, out.y_b_encoded)
        rows.append({
            "method": name,
            "joint_nll": float(joint_log_loss(q, out.y_a_encoded, out.y_b_encoded)),
            "joint_brier": float(joint_brier(q, out.y_a_encoded, out.y_b_encoded)),
            "a_given_b_nll": float(ca),
            "b_given_a_nll": float(cb),
            "marginal_distortion": float(marginal_distortion(q, p.p_a, p.p_b).mean()),
        })

    result = {
        "status": "sanity-only",
        "dataset": "credit",
        "model": "TabICLv2",
        "model_config": {"device": "cpu", "n_estimators": 2, "kv_cache": True, "random_state": 42},
        "fold": 1,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "k_a": int(len(out.classes_a)),
        "k_b": int(len(out.classes_b)),
        "factorization_tv_mean": float(total_variation(j1, j2).mean()),
        "factorization_tv_median": float(np.median(total_variation(j1, j2))),
        "factorization_tv_max": float(total_variation(j1, j2).max()),
        "marginalization_tv_a_mean": float(total_variation(j1.sum(axis=2), p.p_a, axis=1).mean()),
        "marginalization_tv_b_mean": float(total_variation(j2.sum(axis=1), p.p_b, axis=1).mean()),
        "methods": rows,
    }
    out_path = Path("results/real_sanity_credit_tabiclv2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
