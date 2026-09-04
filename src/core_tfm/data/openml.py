from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from .openml_specs import CLASSIFICATION_DATASETS


@dataclass(frozen=True)
class PairDataset:
    name: str
    X: pd.DataFrame
    a: pd.Series
    b: pd.Series


def _drop_missing(X: pd.DataFrame, a: pd.Series, b: pd.Series):
    frame = X.copy()
    frame["__a__"] = np.asarray(a)
    frame["__b__"] = np.asarray(b)
    frame = frame.dropna(axis=0).reset_index(drop=True)
    return frame.drop(columns=["__a__", "__b__"]), frame["__a__"], frame["__b__"]


def _load_wine() -> PairDataset:
    feature_names = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
        "pH", "sulphates", "alcohol",
    ]
    frames = []
    for data_id, color in ((40691, "red"), (40498, "white")):
        ds = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        X = ds.data.copy()
        X.columns = feature_names
        a = pd.Series(ds.target, name="class")
        f = X.copy()
        f["__a__"] = a.to_numpy()
        f["__b__"] = color
        frames.append(f)
    all_data = pd.concat(frames, ignore_index=True)
    all_data = all_data.dropna(axis=0).reset_index(drop=True)
    a = all_data.pop("__a__").astype("category")
    b = all_data.pop("__b__").astype("category")
    return PairDataset("wine", all_data, a, b)


def load_pair_dataset(name: str) -> PairDataset:
    """Load one classification pair from the 2026 consistency benchmark.

    Preprocessing follows Appendix B where practical through sklearn's OpenML
    client: drop missing rows; for MIC drop columns with >2% missing first;
    for Marketing convert the enrollment-date text column to elapsed days.
    Wine is reconstructed by concatenating the pinned red/white source tables.
    """
    key = name.lower()
    if key == "wine":
        return _load_wine()
    if key not in CLASSIFICATION_DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Options: {sorted(CLASSIFICATION_DATASETS)}")

    spec = CLASSIFICATION_DATASETS[key]
    ds = fetch_openml(data_id=spec["openml_id"], as_frame=True, parser="auto")
    X = ds.data.copy()
    a = pd.Series(ds.target, name=spec["target_a"])
    target_b = spec["target_b"]
    if target_b not in X.columns:
        raise KeyError(
            f"Expected second target {target_b!r} in OpenML features for {key}; "
            f"available columns include {list(X.columns)[:10]}."
        )
    b = X.pop(target_b)

    if key == "mic":
        keep = X.columns[X.isna().mean() <= 0.02]
        X = X.loc[:, keep]

    if key == "marketing":
        date_candidates = [
            c for c in X.columns
            if "date" in str(c).lower() or "enroll" in str(c).lower()
        ]
        for col in date_candidates:
            parsed = pd.to_datetime(X[col], errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.95:
                X[col] = (parsed - parsed.min()).dt.days.astype(float)
                break

    X, a, b = _drop_missing(X, a, b)
    a = a.astype("category")
    b = b.astype("category")
    return PairDataset(key, X, a, b)
