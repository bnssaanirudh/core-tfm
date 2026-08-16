"""Validation-gated reconciliation policies.

Selective CoRe treats reconciliation as a decision rather than a mandatory
post-processing step. A policy is selected only from inner-validation
probabilities/labels, then applied unchanged to the held-out test fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np
from numpy.typing import ArrayLike, NDArray

from core_tfm.metrics.scoring import joint_log_loss
from .baselines import arithmetic_pool, geometric_pool
from .mpr import marginal_preserving_reconciliation
from .soft import soft_reconciliation

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SelectivePolicy:
    name: str
    score: float
    weight: float | None = None
    marginal_penalty: float | None = None


@dataclass(frozen=True)
class SelectivePolicySelection:
    policy: SelectivePolicy
    scores: dict[str, float]


def _candidate_key(name: str, weight: float | None = None, lam: float | None = None) -> str:
    if weight is None and lam is None:
        return name
    if lam is None:
        return f"{name}[w={weight:g}]"
    return f"{name}[w={weight:g},lambda={lam:g}]"


def select_reconciliation_policy(j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, y_a: ArrayLike, y_b: ArrayLike, *, weights: Iterable[float] = (0.25, 0.5, 0.75), marginal_penalties: Iterable[float] = (0.1, 1.0, 10.0), min_improvement: float = 0.0) -> SelectivePolicySelection:
    if min_improvement < 0:
        raise ValueError("min_improvement must be non-negative.")
    j1a = np.asarray(j1, dtype=float)
    j2a = np.asarray(j2, dtype=float)
    pa = np.asarray(p_a, dtype=float)
    pb = np.asarray(p_b, dtype=float)
    candidates: list[tuple[SelectivePolicy, FloatArray]] = [
        (SelectivePolicy("j1_b_then_a", np.nan), j1a),
        (SelectivePolicy("j2_a_then_b", np.nan), j2a),
        (SelectivePolicy("arithmetic", np.nan), arithmetic_pool(j1a, j2a)),
        (SelectivePolicy("geometric", np.nan, weight=0.5), geometric_pool(j1a, j2a)),
        (SelectivePolicy("mpr", np.nan, weight=0.5), marginal_preserving_reconciliation(j1a, j2a, pa, pb).joint),
    ]
    for w_raw in weights:
        w = float(w_raw)
        if not 0.0 <= w <= 1.0:
            raise ValueError("All weights must lie in [0, 1].")
        if abs(w - 0.5) > 1e-15:
            candidates.append((SelectivePolicy("geometric", np.nan, weight=w), geometric_pool(j1a, j2a, weight=w)))
            candidates.append((SelectivePolicy("mpr", np.nan, weight=w), marginal_preserving_reconciliation(j1a, j2a, pa, pb, reference_weight=w).joint))
        for lam_raw in marginal_penalties:
            lam = float(lam_raw)
            if lam < 0:
                raise ValueError("Marginal penalties must be non-negative.")
            q = soft_reconciliation(j1a, j2a, pa, pb, reference_weight=w, lambda_a=lam, lambda_b=lam).joint
            candidates.append((SelectivePolicy("soft", np.nan, weight=w, marginal_penalty=lam), q))
    scored: list[tuple[SelectivePolicy, float]] = []
    score_map: dict[str, float] = {}
    for policy, q in candidates:
        s = float(joint_log_loss(q, y_a, y_b))
        key = _candidate_key(policy.name, policy.weight, policy.marginal_penalty)
        score_map[key] = s
        scored.append((policy, s))
    complexity = {"j1_b_then_a": 0, "j2_a_then_b": 0, "arithmetic": 1, "geometric": 2, "mpr": 3, "soft": 4}
    best_policy, best_score = min(scored, key=lambda ps: (ps[1], complexity[ps[0].name], abs((ps[0].weight if ps[0].weight is not None else 0.5) - 0.5), abs(np.log10(max(ps[0].marginal_penalty or 1.0, 1e-12)))))
    original_scored = [(p, s) for p, s in scored if p.name in {"j1_b_then_a", "j2_a_then_b"}]
    best_original, best_original_score = min(original_scored, key=lambda ps: ps[1])
    is_repair = best_policy.name not in {"j1_b_then_a", "j2_a_then_b"}
    if is_repair and best_original_score - best_score < min_improvement:
        best_policy, best_score = best_original, best_original_score
    return SelectivePolicySelection(policy=SelectivePolicy(best_policy.name, float(best_score), best_policy.weight, best_policy.marginal_penalty), scores=score_map)


def apply_reconciliation_policy(selection: SelectivePolicySelection | SelectivePolicy, j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike) -> FloatArray:
    policy = selection.policy if isinstance(selection, SelectivePolicySelection) else selection
    if policy.name == "j1_b_then_a":
        return np.asarray(j1, dtype=float)
    if policy.name == "j2_a_then_b":
        return np.asarray(j2, dtype=float)
    if policy.name == "arithmetic":
        return arithmetic_pool(j1, j2)
    if policy.name == "geometric":
        return geometric_pool(j1, j2, weight=0.5 if policy.weight is None else policy.weight)
    if policy.name == "mpr":
        return marginal_preserving_reconciliation(j1, j2, p_a, p_b, reference_weight=0.5 if policy.weight is None else policy.weight).joint
    if policy.name == "soft":
        return soft_reconciliation(j1, j2, p_a, p_b, reference_weight=0.5 if policy.weight is None else policy.weight, lambda_a=1.0 if policy.marginal_penalty is None else policy.marginal_penalty, lambda_b=1.0 if policy.marginal_penalty is None else policy.marginal_penalty).joint
    raise ValueError(f"Unknown policy: {policy.name}")
