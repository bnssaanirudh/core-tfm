from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np
from numpy.typing import ArrayLike

from .baselines import geometric_pool
from .mpr import marginal_preserving_reconciliation
from .soft import soft_reconciliation
from core_tfm.metrics.scoring import joint_log_loss


@dataclass(frozen=True)
class WeightSelection:
    weight: float
    scores: dict[float, float]


@dataclass(frozen=True)
class SoftHyperparameterSelection:
    weight: float
    marginal_penalty: float
    score: float
    scores: dict[tuple[float, float], float]


def select_reference_weight(j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, y_a: ArrayLike, y_b: ArrayLike, *, candidates: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> WeightSelection:
    scores: dict[float, float] = {}
    for w in candidates:
        w = float(w)
        if not 0.0 <= w <= 1.0:
            raise ValueError("All candidate weights must lie in [0, 1].")
        q = marginal_preserving_reconciliation(j1, j2, p_a, p_b, reference_weight=w).joint
        scores[w] = joint_log_loss(q, y_a, y_b)
    best = min(scores, key=lambda w: (scores[w], abs(w - 0.5)))
    return WeightSelection(weight=float(best), scores=scores)


def select_geometric_pool_weight(j1: ArrayLike, j2: ArrayLike, y_a: ArrayLike, y_b: ArrayLike, *, candidates: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> WeightSelection:
    scores: dict[float, float] = {}
    for w in candidates:
        w = float(w)
        if not 0.0 <= w <= 1.0:
            raise ValueError("All candidate weights must lie in [0, 1].")
        q = geometric_pool(j1, j2, weight=w)
        scores[w] = joint_log_loss(q, y_a, y_b)
    best = min(scores, key=lambda w: (scores[w], abs(w - 0.5)))
    return WeightSelection(weight=float(best), scores=scores)


def select_soft_hyperparameters(j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, y_a: ArrayLike, y_b: ArrayLike, *, weights: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0), marginal_penalties: Iterable[float] = (0.1, 1.0, 10.0)) -> SoftHyperparameterSelection:
    scores: dict[tuple[float, float], float] = {}
    for w in weights:
        w = float(w)
        if not 0.0 <= w <= 1.0:
            raise ValueError("All weights must lie in [0, 1].")
        for lam in marginal_penalties:
            lam = float(lam)
            if lam < 0:
                raise ValueError("Marginal penalties must be non-negative.")
            q = soft_reconciliation(j1, j2, p_a, p_b, reference_weight=w, lambda_a=lam, lambda_b=lam).joint
            scores[(w, lam)] = joint_log_loss(q, y_a, y_b)
    best = min(scores, key=lambda pair: (scores[pair], abs(pair[0] - 0.5), abs(np.log10(max(pair[1], 1e-12)))))
    return SoftHyperparameterSelection(weight=float(best[0]), marginal_penalty=float(best[1]), score=float(scores[best]), scores=scores)


@dataclass(frozen=True)
class OrderSelection:
    name: str
    score: float
    scores: dict[str, float]


def select_original_order(j1: ArrayLike, j2: ArrayLike, y_a: ArrayLike, y_b: ArrayLike) -> OrderSelection:
    scores = {"j1_b_then_a": joint_log_loss(j1, y_a, y_b), "j2_a_then_b": joint_log_loss(j2, y_a, y_b)}
    name = min(scores, key=scores.get)
    return OrderSelection(name=name, score=float(scores[name]), scores={k: float(v) for k, v in scores.items()})
