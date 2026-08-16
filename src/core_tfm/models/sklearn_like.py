from __future__ import annotations

from typing import Callable, Any
import numpy as np
import pandas as pd

from .base import ProbabilisticClassifierAdapter


class SklearnLikeAdapter(ProbabilisticClassifierAdapter):
    def __init__(self, constructor: Callable[[], Any]):
        self._constructor = constructor
        self._model = constructor()

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "SklearnLikeAdapter":
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self._model.classes_)
