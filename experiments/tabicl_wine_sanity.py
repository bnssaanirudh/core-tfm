"""One-fold TabICLv2 sanity benchmark on the source-paper Wine reconstruction.

Klötergens et al. (2026) concatenate OpenML red-wine ID 40691 and white-wine
ID 40498, add source-table color as B, keep red quality at raw values 3--8,
and use index-encoded white quality values 1--7. OpenML's metadata API is
currently returning HTTP 504 from GitHub-hosted runners, so this script
reconstructs the same representation from the canonical UCI Wine Quality files:
red quality is unchanged and white quality is transformed as quality - 2.
The UCI archive and member hashes are recorded in the result.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from core_tfm.data.openml import PairDataset
from core_tfm.inference.extract import extract_pair_predictions
from core_tfm.metrics.distributions import marginal_distortion, total_variation
from core_tfm.metrics.scoring import joint_brier, joint_log_loss
from core_tfm.models.tfm_adapters import tabiclv2_adapter
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool, independent_joint
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.soft import soft_reconciliation

CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_source_equivalent_wine() -> tuple[PairDataset, dict]:
    with urllib.request.urlopen(UCI_ZIP, timeout=60) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        red_bytes = zf.read("winequality-red.csv")
        white_bytes = zf.read("winequality-white.csv")

    red = pd.read_csv(io.BytesIO(red_bytes), sep=";")
    white = pd.read_csv(io.BytesIO(white_bytes), sep=";")
    if len(red) != 1599 or len(white) != 4898:
        raise RuntimeError(f"Unexpected UCI Wine row counts: red={len(red)}, white={len(white)}")
    if list(red.columns) != list(white.columns):
        raise RuntimeError("Red/white UCI Wine schemas differ unexpectedly")
    if "quality" not in red.columns:
        raise RuntimeError("Expected UCI quality column is missing")

    red_x = red.drop(columns=["quality"]).copy()
    white_x = white.drop(columns=["quality"]).copy()
    red_a = red["quality"].astype(int)
    # Source Appendix B: the white OpenML table stores its seven ordered quality
    # levels (raw 3--9) as indices 1--7.
    white_a = white["quality"].astype(int) - 2
    if sorted(red_a.unique().tolist()) != [3, 4, 5, 6, 7, 8]:
        raise RuntimeError("Unexpected red-wine quality levels")
    if sorted(white_a.unique().tolist()) != [1, 2, 3, 4, 5, 6, 7]:
        raise RuntimeError("Unexpected encoded white-wine quality levels")

    X = pd.concat([red_x, white_x], ignore_index=True)
    a = pd.concat([red_a, white_a], ignore_index=True).astype("category")
    b = pd.Series(
        ["red"] * len(red) + ["white"] * len(white), name="color", dtype="category"
    )
    provenance = {
        "route": "source-equivalent reconstruction from canonical UCI files",
        "canonical_source": UCI_ZIP,
        "source_paper_openml_ids": [40691, 40498],
        "transformation": "red quality unchanged; white quality = raw quality - 2; add color",
        "archive_sha256": _sha256(archive),
        "red_csv_sha256": _sha256(red_bytes),
        "white_csv_sha256": _sha256(white_bytes),
        "red_rows": int(len(red)),
        "white_rows": int(len(white)),
    }
    return PairDataset("wine", X, a, b), provenance


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
    data, provenance = load_source_equivalent_wine()
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
        "dataset": "Wine (source-equivalent OpenML 40691 + 40498 reconstruction)",
        "data_provenance": provenance,
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
