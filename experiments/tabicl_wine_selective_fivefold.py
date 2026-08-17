"""Five-fold fixed + leakage-free Selective CoRe benchmark on TabICLv2 Wine.

The Wine table is reconstructed from the canonical UCI red/white CSVs exactly as
described in Klötergens et al. (2026): red quality remains 3--8, white raw
quality 3--9 is index-encoded as 1--7 (quality - 2), and source-table color is
added as the second target. This avoids the currently failing OpenML metadata API
while preserving the stated OpenML 40691 + 40498 representation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from core_tfm.inference.extract import extract_pair_predictions
from core_tfm.metrics.distributions import marginal_distortion, total_variation
from core_tfm.metrics.scoring import joint_brier, joint_log_loss
from core_tfm.models.tfm_adapters import tabiclv2_adapter
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool, independent_joint
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.selective import apply_reconciliation_policy, select_reconciliation_policy
from core_tfm.reconciliation.soft import soft_reconciliation

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PENALTIES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_wine():
    with urllib.request.urlopen(UCI_ZIP, timeout=60) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        red_bytes = zf.read("winequality-red.csv")
        white_bytes = zf.read("winequality-white.csv")
    red = pd.read_csv(io.BytesIO(red_bytes), sep=";")
    white = pd.read_csv(io.BytesIO(white_bytes), sep=";")
    if (len(red), len(white)) != (1599, 4898):
        raise RuntimeError(f"Unexpected Wine row counts: {len(red)}, {len(white)}")
    if list(red.columns) != list(white.columns) or "quality" not in red.columns:
        raise RuntimeError("Unexpected Wine schema")
    red_a = red.pop("quality").astype(int)
    white_a = white.pop("quality").astype(int) - 2
    if sorted(red_a.unique()) != [3, 4, 5, 6, 7, 8]:
        raise RuntimeError("Unexpected red target levels")
    if sorted(white_a.unique()) != [1, 2, 3, 4, 5, 6, 7]:
        raise RuntimeError("Unexpected encoded white target levels")
    X = pd.concat([red, white], ignore_index=True)
    a = pd.concat([red_a, white_a], ignore_index=True).astype("category")
    b = pd.Series(["red"] * len(red) + ["white"] * len(white), dtype="category")
    provenance = {
        "route": "source-equivalent reconstruction from canonical UCI files",
        "canonical_source": UCI_ZIP,
        "source_paper_openml_ids": [40691, 40498],
        "transformation": "red quality unchanged; white quality = raw quality - 2; add color",
        "archive_sha256": _sha256(archive),
        "red_csv_sha256": _sha256(red_bytes),
        "white_csv_sha256": _sha256(white_bytes),
        "red_rows": len(red),
        "white_rows": len(white),
    }
    return X, a, b, provenance


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


def extract(X, a, b, train_idx, eval_idx):
    return extract_pair_predictions(
        factory,
        X.iloc[train_idx].reset_index(drop=True),
        a.iloc[train_idx].reset_index(drop=True),
        b.iloc[train_idx].reset_index(drop=True),
        X.iloc[eval_idx].reset_index(drop=True),
        a_test=a.iloc[eval_idx].reset_index(drop=True),
        b_test=b.iloc[eval_idx].reset_index(drop=True),
    )


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


def policy_dict(policy):
    return {
        "name": policy.name,
        "weight": None if policy.weight is None else float(policy.weight),
        "marginal_penalty": None if policy.marginal_penalty is None else float(policy.marginal_penalty),
        "validation_joint_nll": float(policy.score),
    }


def main():
    X, a, b, provenance = load_wine()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    method_values = defaultdict(lambda: defaultdict(list))

    for fold, (outer_train_idx, test_idx) in enumerate(outer.split(X, a), start=1):
        inner_train_idx, val_idx = train_test_split(
            outer_train_idx,
            test_size=0.20,
            random_state=2000 + fold,
            stratify=a.iloc[outer_train_idx],
        )
        val = extract(X, a, b, inner_train_idx, val_idx)
        vp = val.predictions
        selection = select_reconciliation_policy(
            vp.j_b_then_a,
            vp.j_a_then_b,
            vp.p_a,
            vp.p_b,
            val.y_a_encoded,
            val.y_b_encoded,
            weights=WEIGHTS,
            marginal_penalties=PENALTIES,
        )

        test = extract(X, a, b, outer_train_idx, test_idx)
        p = test.predictions
        j1, j2 = p.j_b_then_a, p.j_a_then_b
        selected = apply_reconciliation_policy(selection, j1, j2, p.p_a, p.p_b)
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
            "selective_core": selected,
        }
        fold_methods = []
        for name, q in methods.items():
            nll = float(joint_log_loss(q, test.y_a_encoded, test.y_b_encoded))
            brier = float(joint_brier(q, test.y_a_encoded, test.y_b_encoded))
            md = float(marginal_distortion(q, p.p_a, p.p_b).mean())
            fold_methods.append({"method": name, "joint_nll": nll, "joint_brier": brier, "marginal_distortion": md})
            method_values[name]["joint_nll"].append(nll)
            method_values[name]["joint_brier"].append(brier)
            method_values[name]["marginal_distortion"].append(md)

        j1_nll = next(x["joint_nll"] for x in fold_methods if x["method"] == "j1_b_then_a")
        j2_nll = next(x["joint_nll"] for x in fold_methods if x["method"] == "j2_a_then_b")
        sel_nll = next(x["joint_nll"] for x in fold_methods if x["method"] == "selective_core")
        tv = total_variation(j1, j2)
        folds.append(
            {
                "fold": fold,
                "n_outer_train": int(len(outer_train_idx)),
                "n_inner_train": int(len(inner_train_idx)),
                "n_validation": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "selected_policy": policy_dict(selection.policy),
                "factorization_tv_mean": float(tv.mean()),
                "factorization_tv_median": float(np.median(tv)),
                "methods": fold_methods,
                "best_original_joint_nll": min(j1_nll, j2_nll),
                "selected_minus_best_original_nll": sel_nll - min(j1_nll, j2_nll),
            }
        )

    method_summary = {}
    for name, metrics in method_values.items():
        method_summary[name] = {metric: mean_std(values) for metric, values in metrics.items()}
    policy_counts = Counter(f["selected_policy"]["name"] for f in folds)
    aggregate = {
        "factorization_tv": mean_std([f["factorization_tv_mean"] for f in folds]),
        "methods": method_summary,
        "best_original_joint_nll": mean_std([f["best_original_joint_nll"] for f in folds]),
        "selective_minus_best_original_nll": mean_std([f["selected_minus_best_original_nll"] for f in folds]),
        "policy_name_counts": dict(sorted(policy_counts.items())),
        "selected_original_folds": int(sum(f["selected_policy"]["name"] in {"j1_b_then_a", "j2_a_then_b"} for f in folds)),
    }

    result = {
        "status": "small-real-selective-benchmark",
        "dataset": "Wine (source-equivalent OpenML 40691 + 40498 reconstruction)",
        "data_provenance": provenance,
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
        "source_paper_tabiclv2_factorization_tv": {"mean": 0.0193, "fold_std": 0.0007},
        "outer_split": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "selection": {
            "validation_fraction": 0.20,
            "validation_seed_rule": "2000 + outer_fold",
            "weights": list(WEIGHTS),
            "marginal_penalties": list(PENALTIES),
            "test_labels_used_for_selection": False,
        },
        "folds": folds,
        "aggregate": aggregate,
    }
    path = Path("results/real_wine_tabiclv2_selective_fivefold.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
