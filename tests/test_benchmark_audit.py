import numpy as np
import pandas as pd
import pytest

from core_tfm.benchmark_audit import audit_fold_matrix, dataset_blocked_effect


def small_frame() -> pd.DataFrame:
    rows = []
    for dataset in ("d1", "d2"):
        for model in ("m1", "m2"):
            for fold in (1, 2):
                for method, offset in (("arithmetic", 0.0), ("selective_core", 0.1)):
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "fold": fold,
                            "method": method,
                            "joint_nll": fold + offset,
                            "joint_brier": 0.2 + offset,
                            "joint_ece_15": 0.1,
                            "marginal_distortion": 0.05,
                            "factorization_tv": 0.03,
                        }
                    )
    return pd.DataFrame(rows)


def test_audit_fold_matrix_accepts_complete_matrix():
    result = audit_fold_matrix(
        small_frame(), expected_datasets=2, expected_models=2, expected_folds=2, expected_methods=2
    )
    assert result.rows == 16
    assert result.cells == 4
    assert result.fold_tasks == 8


def test_audit_fold_matrix_rejects_duplicates_and_nonfinite_metrics():
    frame = small_frame()
    with pytest.raises(ValueError, match="duplicate"):
        audit_fold_matrix(
            pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            expected_datasets=2,
            expected_models=2,
            expected_folds=2,
            expected_methods=2,
        )
    frame.loc[0, "joint_nll"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        audit_fold_matrix(
            frame, expected_datasets=2, expected_models=2, expected_folds=2, expected_methods=2
        )


def test_dataset_blocked_effect_uses_datasets_as_units():
    effects, summary = dataset_blocked_effect(
        small_frame(),
        metric="joint_nll",
        method="selective_core",
        comparator="arithmetic",
        primary_models=("m1", "m2"),
        n_boot=500,
        seed=1,
    )
    assert list(effects.index) == ["d1", "d2"]
    assert np.allclose(effects.to_numpy(), 0.1)
    assert summary["n_datasets"] == 2
    assert summary["wins"] == 0
    assert summary["losses"] == 2

