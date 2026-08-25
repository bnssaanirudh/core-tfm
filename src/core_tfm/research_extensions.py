"""Research utilities for reliability-aware CoRe-TFM experiments.

These helpers are intentionally model-agnostic. They operate on probability
arrays or benchmark result tables and are designed for the post-hoc analyses
introduced in the reliability-aware research extension notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OracleDecomposition:
    arithmetic_loss: float
    selected_loss: float
    oracle_loss: float
    oracle_method: str
    opportunity: float
    selection_regret: float
    selected_vs_arithmetic: float


def total_variation(p: ArrayLike, q: ArrayLike) -> FloatArray:
    """Per-example total variation distance over all non-batch axes."""
    pa = np.asarray(p, dtype=float)
    qa = np.asarray(q, dtype=float)
    if pa.shape != qa.shape:
        raise ValueError("p and q must have identical shapes")
    if pa.ndim < 2:
        raise ValueError("Expected a batch axis plus at least one probability axis")
    axes = tuple(range(1, pa.ndim))
    return 0.5 * np.sum(np.abs(pa - qa), axis=axes)


def predictive_entropy(p: ArrayLike, eps: float = 1e-15) -> FloatArray:
    """Entropy of categorical predictions; accepts (..., K)."""
    pa = np.asarray(p, dtype=float)
    pa = np.maximum(pa, eps)
    pa = pa / pa.sum(axis=-1, keepdims=True)
    return -np.sum(pa * np.log(pa), axis=-1)


def audit_probability_views(j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike) -> dict[str, FloatArray | float]:
    """Lightweight consistency audit for four-view probability outputs.

    Returns per-example factorization TV, marginalization defects, and entropy
    diagnostics plus their means. This is descriptive only: the audit does not
    claim that high inconsistency implies poor predictive accuracy.
    """
    j1a = np.asarray(j1, dtype=float)
    j2a = np.asarray(j2, dtype=float)
    pa = np.asarray(p_a, dtype=float)
    pb = np.asarray(p_b, dtype=float)
    if j1a.shape != j2a.shape or j1a.ndim != 3:
        raise ValueError("j1 and j2 must share shape (n, k_a, k_b)")
    if pa.shape != (j1a.shape[0], j1a.shape[1]):
        raise ValueError("p_a must have shape (n, k_a)")
    if pb.shape != (j1a.shape[0], j1a.shape[2]):
        raise ValueError("p_b must have shape (n, k_b)")
    factor_tv = total_variation(j1a, j2a)
    implied_a_from_j1 = j1a.sum(axis=2)
    implied_b_from_j2 = j2a.sum(axis=1)
    defect_a = total_variation(implied_a_from_j1, pa)
    defect_b = total_variation(implied_b_from_j2, pb)
    return {
        "factorization_tv": factor_tv,
        "marginalization_defect_a": defect_a,
        "marginalization_defect_b": defect_b,
        "entropy_direct_a": predictive_entropy(pa),
        "entropy_direct_b": predictive_entropy(pb),
        "mean_factorization_tv": float(np.mean(factor_tv)),
        "mean_marginalization_defect_a": float(np.mean(defect_a)),
        "mean_marginalization_defect_b": float(np.mean(defect_b)),
    }


def jensen_shannon_dispersion(predictions: ArrayLike, eps: float = 1e-15) -> FloatArray:
    """Per-example dispersion across repeated probability predictions.

    Parameters
    ----------
    predictions:
        Array shaped (n_members, n_examples, n_classes).
    """
    x = np.asarray(predictions, dtype=float)
    if x.ndim != 3:
        raise ValueError("predictions must have shape (members, examples, classes)")
    x = np.maximum(x, eps)
    x = x / x.sum(axis=-1, keepdims=True)
    mean = x.mean(axis=0)
    h_mean = predictive_entropy(mean, eps=eps)
    mean_h = predictive_entropy(x, eps=eps).mean(axis=0)
    return np.asarray(h_mean - mean_h, dtype=float)


def inverse_dispersion_weight(d1: ArrayLike, d2: ArrayLike, temperature: float = 1.0) -> FloatArray:
    """Convert two instability signals into a per-example weight on view 1."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    a = np.asarray(d1, dtype=float)
    b = np.asarray(d2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("dispersion arrays must have the same shape")
    z1 = -a / temperature
    z2 = -b / temperature
    m = np.maximum(z1, z2)
    e1 = np.exp(z1 - m)
    e2 = np.exp(z2 - m)
    return e1 / (e1 + e2)


def support_adaptive_penalties(counts: ArrayLike, base_lambda: float = 10.0, tau: float = 10.0) -> FloatArray:
    """Simple rare-class-aware marginal penalties.

    lambda_c = base_lambda * n_c / (n_c + tau)
    """
    if base_lambda < 0 or tau <= 0:
        raise ValueError("base_lambda must be non-negative and tau positive")
    n = np.asarray(counts, dtype=float)
    if np.any(n < 0):
        raise ValueError("counts must be non-negative")
    return base_lambda * n / (n + tau)


def complexity_penalty(n_validation: int, family_size: int, delta: float = 0.05, beta: float = 1.0) -> float:
    """SRM-style validation complexity penalty used by Safe Selective CoRe."""
    if n_validation <= 0 or family_size <= 0:
        raise ValueError("n_validation and family_size must be positive")
    if not 0 < delta < 1 or beta < 0:
        raise ValueError("delta must be in (0,1) and beta non-negative")
    return beta * math.sqrt((math.log(family_size) + math.log(1.0 / delta)) / (2.0 * n_validation))


def safe_select_from_scores(
    family_scores: Mapping[str, Mapping[str, float]],
    *,
    n_validation: int,
    fallback: str = "arithmetic",
    delta: float = 0.05,
    beta: float = 1.0,
) -> tuple[str, str, dict[str, float]]:
    """Structural-risk-style selection over nested candidate families.

    family_scores maps family name -> {candidate -> validation loss}. The
    function chooses the family minimizing best empirical loss + complexity
    penalty, then returns that family's best candidate. If the winning family
    does not improve on the fallback by more than its complexity penalty, the
    fallback is returned.
    """
    if not family_scores:
        raise ValueError("At least one family is required")
    diagnostics: dict[str, float] = {}
    best_family = None
    best_candidate = None
    best_criterion = float("inf")
    fallback_score = None
    for family, scores in family_scores.items():
        if not scores:
            continue
        if fallback in scores and fallback_score is None:
            fallback_score = float(scores[fallback])
        candidate, loss = min(scores.items(), key=lambda kv: kv[1])
        pen = complexity_penalty(n_validation, len(scores), delta=delta, beta=beta)
        criterion = float(loss) + pen
        diagnostics[f"{family}.best_loss"] = float(loss)
        diagnostics[f"{family}.penalty"] = pen
        diagnostics[f"{family}.criterion"] = criterion
        if criterion < best_criterion:
            best_criterion = criterion
            best_family = family
            best_candidate = candidate
    if best_family is None or best_candidate is None:
        raise ValueError("No non-empty family was provided")
    if fallback_score is not None:
        winning_penalty = diagnostics[f"{best_family}.penalty"]
        winning_loss = diagnostics[f"{best_family}.best_loss"]
        if fallback_score - winning_loss <= winning_penalty:
            return fallback, "fallback", diagnostics
    return best_candidate, best_family, diagnostics


def oracle_selection_decomposition(
    losses: Mapping[str, float],
    *,
    selected_method: str,
    arithmetic_method: str = "arithmetic",
) -> OracleDecomposition:
    """Decompose a selector's loss into available opportunity and selection regret."""
    if selected_method not in losses or arithmetic_method not in losses:
        raise KeyError("selected_method and arithmetic_method must be present")
    oracle_method, oracle_loss = min(losses.items(), key=lambda kv: kv[1])
    arithmetic_loss = float(losses[arithmetic_method])
    selected_loss = float(losses[selected_method])
    oracle_loss = float(oracle_loss)
    return OracleDecomposition(
        arithmetic_loss=arithmetic_loss,
        selected_loss=selected_loss,
        oracle_loss=oracle_loss,
        oracle_method=oracle_method,
        opportunity=arithmetic_loss - oracle_loss,
        selection_regret=selected_loss - oracle_loss,
        selected_vs_arithmetic=selected_loss - arithmetic_loss,
    )


def decision_regret(
    q: ArrayLike,
    p_true: ArrayLike,
    utility: ArrayLike,
) -> FloatArray:
    """Expected decision regret for a discrete joint distribution.

    q and p_true have shape (n, k_a, k_b); utility has shape
    (n_actions, k_a, k_b). Returned values are per-example regret under p_true.
    """
    qa = np.asarray(q, dtype=float)
    pt = np.asarray(p_true, dtype=float)
    u = np.asarray(utility, dtype=float)
    if qa.shape != pt.shape or qa.ndim != 3:
        raise ValueError("q and p_true must share shape (n, k_a, k_b)")
    if u.ndim != 3 or u.shape[1:] != qa.shape[1:]:
        raise ValueError("utility must have shape (actions, k_a, k_b)")
    pred_eu = np.einsum("nab,kab->nk", qa, u)
    true_eu = np.einsum("nab,kab->nk", pt, u)
    chosen = pred_eu.argmax(axis=1)
    optimal = true_eu.max(axis=1)
    obtained = true_eu[np.arange(len(qa)), chosen]
    return optimal - obtained


def inconsistency_accuracy_counterexamples(eps_values: Sequence[float] | None = None) -> dict[str, list[dict[str, float]]]:
    """Construct two analytic families showing TV is not an accuracy signal.

    Family A: J1 is the truth and J2 moves mass away from the true cell. TV can
    become large while arithmetic reconciliation necessarily hurts NLL.

    Family B: both views move symmetrically around the truth. TV can be nonzero
    while arithmetic pooling recovers the truth exactly, so repair gain can be
    positive even for arbitrarily small inconsistency.
    """
    values = list(eps_values or (1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.4))
    out = {"large_tv_no_need_to_repair": [], "small_tv_repair_can_help": []}
    truth = np.array([[0.7, 0.1], [0.1, 0.1]], dtype=float)
    for e in values:
        e = float(e)
        d = min(e, 0.69)
        j1 = truth.copy()
        j2 = truth.copy()
        j2[0, 0] -= d
        j2[1, 1] += d
        arithmetic = 0.5 * (j1 + j2)
        tv = 0.5 * np.abs(j1 - j2).sum()
        nll_truth = -math.log(truth[0, 0])
        nll_arith = -math.log(arithmetic[0, 0])
        out["large_tv_no_need_to_repair"].append({"epsilon": e, "tv": tv, "arithmetic_regret": nll_arith - nll_truth})

        d = min(e, 0.09)
        j1 = truth.copy(); j2 = truth.copy()
        j1[0, 0] -= d; j1[0, 1] += d
        j2[0, 0] += d; j2[0, 1] -= d
        arithmetic = 0.5 * (j1 + j2)
        tv = 0.5 * np.abs(j1 - j2).sum()
        raw_regret = -math.log(j1[0, 0]) + math.log(truth[0, 0])
        repaired_regret = -math.log(arithmetic[0, 0]) + math.log(truth[0, 0])
        out["small_tv_repair_can_help"].append({"epsilon": e, "tv": tv, "raw_regret": raw_regret, "arithmetic_regret": repaired_regret})
    return out
