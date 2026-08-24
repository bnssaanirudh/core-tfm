from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core_tfm.metrics.statistics import paired_bootstrap_mean_difference, paired_wilcoxon


REQUIRED_METRICS = (
    "joint_nll",
    "joint_brier",
    "joint_ece_15",
    "marginal_distortion",
    "factorization_tv",
)


@dataclass(frozen=True)
class MatrixAudit:
    rows: int
    cells: int
    fold_tasks: int
    datasets: int
    models: int
    methods_per_fold: int


def audit_fold_matrix(
    frame: pd.DataFrame,
    *,
    expected_datasets: int,
    expected_models: int,
    expected_folds: int,
    expected_methods: int,
) -> MatrixAudit:
    """Validate the structural and numerical integrity of a fold-level benchmark."""
    required = {"dataset", "model", "fold", "method", *REQUIRED_METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing benchmark columns: {sorted(missing)}")

    keys = ["dataset", "model", "fold", "method"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate dataset-model-fold-method rows")
    if not np.isfinite(frame[list(REQUIRED_METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("non-finite benchmark metric")

    task_counts = frame.groupby(["dataset", "model", "fold"], sort=False).size()
    if set(task_counts.unique()) != {expected_methods}:
        raise ValueError("every fold task must contain the expected method count")
    cell_folds = frame.groupby(["dataset", "model"], sort=False)["fold"].nunique()
    if set(cell_folds.unique()) != {expected_folds}:
        raise ValueError("every dataset-model cell must contain the expected fold count")
    if frame["dataset"].nunique() != expected_datasets:
        raise ValueError("unexpected dataset count")
    if frame["model"].nunique() != expected_models:
        raise ValueError("unexpected model count")

    return MatrixAudit(
        rows=len(frame),
        cells=len(cell_folds),
        fold_tasks=len(task_counts),
        datasets=frame["dataset"].nunique(),
        models=frame["model"].nunique(),
        methods_per_fold=expected_methods,
    )


def dataset_blocked_effect(
    frame: pd.DataFrame,
    *,
    metric: str,
    method: str,
    comparator: str,
    primary_models: tuple[str, ...],
    n_boot: int = 100_000,
    seed: int = 20260823,
) -> tuple[pd.Series, dict[str, float | int]]:
    """Compute a paired effect with datasets, not rows or model cells, as units.

    Fold means are first formed within each dataset-model-method cell. Effects are
    then averaged over the explicitly supplied primary models within each dataset.
    The returned sign is ``method - comparator``; negative values favor ``method``.
    """
    if metric not in frame:
        raise ValueError(f"unknown metric: {metric}")
    selected = frame.loc[frame["model"].isin(primary_models)]
    present = set(selected["model"].unique())
    if present != set(primary_models):
        raise ValueError(f"primary model mismatch: expected {primary_models}, found {sorted(present)}")

    means = selected.groupby(["dataset", "model", "method"], sort=True)[metric].mean().unstack()
    for name in (method, comparator):
        if name not in means:
            raise ValueError(f"method not found: {name}")
    effect = (means[method] - means[comparator]).groupby("dataset").mean().sort_index()
    zeros = np.zeros(len(effect), dtype=float)
    mean, low, high = paired_bootstrap_mean_difference(
        effect.to_numpy(), zeros, n_boot=n_boot, seed=seed
    )
    _, p_value = paired_wilcoxon(effect.to_numpy(), zeros)
    summary: dict[str, float | int] = {
        "n_datasets": len(effect),
        "mean_difference": mean,
        "bootstrap_95_ci_low": low,
        "bootstrap_95_ci_high": high,
        "wins": int((effect < 0).sum()),
        "losses": int((effect > 0).sum()),
        "ties": int((effect == 0).sum()),
        "wilcoxon_two_sided_p": p_value,
    }
    return effect, summary

