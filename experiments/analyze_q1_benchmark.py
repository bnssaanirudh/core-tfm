from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core_tfm.benchmark_audit import audit_fold_matrix, dataset_blocked_effect


DEFAULT_RESULTS = Path("results/q1_fast_complete_256_v1")
PRIMARY_MODELS = ("tabiclv2", "tabpfn3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the bounded CoRe-TFM benchmark.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    folds = pd.read_csv(args.results / "fold_results.csv")
    audit = audit_fold_matrix(
        folds,
        expected_datasets=10,
        expected_models=3,
        expected_folds=5,
        expected_methods=8,
    )
    analyses = {}
    for metric in ("joint_nll", "joint_brier", "joint_ece_15", "marginal_distortion"):
        effect, summary = dataset_blocked_effect(
            folds,
            metric=metric,
            method="selective_core",
            comparator="arithmetic",
            primary_models=PRIMARY_MODELS,
        )
        analyses[metric] = {"by_dataset": effect.to_dict(), **summary}

    print(json.dumps({"matrix": audit.__dict__, "selective_minus_arithmetic": analyses}, indent=2))


if __name__ == "__main__":
    main()

