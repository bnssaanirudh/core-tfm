from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd

from core_tfm.models.base import ProbabilisticClassifierAdapter
from core_tfm.inference.joints import JointPredictions, construct_implied_joints


@dataclass(frozen=True)
class ExtractedPair:
    predictions: JointPredictions
    classes_a: np.ndarray
    classes_b: np.ndarray
    y_a_encoded: np.ndarray | None = None
    y_b_encoded: np.ndarray | None = None


def _append_condition_column(X: pd.DataFrame, name: str, value, train_series: pd.Series) -> pd.DataFrame:
    out = X.copy()
    if isinstance(train_series.dtype, pd.CategoricalDtype):
        out[name] = pd.Categorical([value] * len(out), categories=train_series.cat.categories)
    else:
        out[name] = value
        try:
            out[name] = out[name].astype(train_series.dtype)
        except (TypeError, ValueError):
            pass
    return out


def _encode_with_classes(values: pd.Series | np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {v: i for i, v in enumerate(classes.tolist())}
    try:
        return np.array([lookup[v] for v in np.asarray(values)], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Observed label {exc.args[0]!r} not found in model classes.") from exc


def extract_pair_predictions(model_factory: Callable[[], ProbabilisticClassifierAdapter], X_train: pd.DataFrame, a_train: pd.Series, b_train: pd.Series, X_test: pd.DataFrame, *, a_test: pd.Series | None = None, b_test: pd.Series | None = None, condition_a_name: str = "__target_a__", condition_b_name: str = "__target_b__") -> ExtractedPair:
    if condition_a_name in X_train.columns or condition_b_name in X_train.columns:
        raise ValueError("Condition-column names collide with existing features.")
    model_a = model_factory().fit(X_train, a_train)
    p_a = model_a.predict_proba(X_test)
    classes_a = model_a.classes_
    model_b = model_factory().fit(X_train, b_train)
    p_b = model_b.predict_proba(X_test)
    classes_b = model_b.classes_
    xab_train = X_train.copy()
    xab_train[condition_b_name] = b_train.to_numpy()
    if isinstance(b_train.dtype, pd.CategoricalDtype):
        xab_train[condition_b_name] = pd.Categorical(xab_train[condition_b_name], categories=b_train.cat.categories)
    model_a_given_b = model_factory().fit(xab_train, a_train)
    if not np.array_equal(model_a_given_b.classes_, classes_a):
        raise ValueError("A class order differs between direct and conditional models.")
    p_a_given_b = np.empty((len(X_test), len(classes_b), len(classes_a)), dtype=float)
    for j, b_val in enumerate(classes_b):
        xb = _append_condition_column(X_test, condition_b_name, b_val, b_train)
        p_a_given_b[:, j, :] = model_a_given_b.predict_proba(xb)
    xba_train = X_train.copy()
    xba_train[condition_a_name] = a_train.to_numpy()
    if isinstance(a_train.dtype, pd.CategoricalDtype):
        xba_train[condition_a_name] = pd.Categorical(xba_train[condition_a_name], categories=a_train.cat.categories)
    model_b_given_a = model_factory().fit(xba_train, b_train)
    if not np.array_equal(model_b_given_a.classes_, classes_b):
        raise ValueError("B class order differs between direct and conditional models.")
    p_b_given_a = np.empty((len(X_test), len(classes_a), len(classes_b)), dtype=float)
    for i, a_val in enumerate(classes_a):
        xa = _append_condition_column(X_test, condition_a_name, a_val, a_train)
        p_b_given_a[:, i, :] = model_b_given_a.predict_proba(xa)
    predictions = construct_implied_joints(p_a, p_b, p_a_given_b, p_b_given_a)
    ya = _encode_with_classes(a_test, classes_a) if a_test is not None else None
    yb = _encode_with_classes(b_test, classes_b) if b_test is not None else None
    return ExtractedPair(predictions, classes_a, classes_b, ya, yb)
