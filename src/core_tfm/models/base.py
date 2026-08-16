from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd


class ProbabilisticClassifierAdapter(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "ProbabilisticClassifierAdapter": ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray: ...


def clone_adapter(factory: Any) -> ProbabilisticClassifierAdapter:
    model = factory()
    if not isinstance(model, ProbabilisticClassifierAdapter):
        raise TypeError("factory must return a ProbabilisticClassifierAdapter.")
    return model
