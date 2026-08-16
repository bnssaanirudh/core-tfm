from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .baselines import geometric_pool

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SoftReconciliationResult:
    joint: FloatArray
    converged: NDArray[np.bool_]
    iterations: NDArray[np.int64]
    objective_initial: FloatArray
    objective_final: FloatArray
    max_update: FloatArray


def _prepare_batched(x: ArrayLike, ndim_single: int) -> tuple[FloatArray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == ndim_single:
        return arr[None, ...], True
    if arr.ndim == ndim_single + 1:
        return arr, False
    raise ValueError(f"Expected {ndim_single}-D or {ndim_single + 1}-D array, got {arr.ndim}-D.")


def _normalize_rows(p: FloatArray, eps: float) -> FloatArray:
    if np.any(~np.isfinite(p)) or np.any(p < -1e-12):
        raise ValueError("Probabilities must be finite and non-negative.")
    p = np.maximum(p, eps)
    return p / p.sum(axis=1, keepdims=True)


def _kl_per_sample(p: FloatArray, q: FloatArray, axis: tuple[int, ...], eps: float) -> FloatArray:
    p_safe = np.maximum(p, eps)
    q_safe = np.maximum(q, eps)
    return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=axis)


def soft_reconciliation_objective(q: ArrayLike, reference: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, *, lambda_a: float = 1.0, lambda_b: float = 1.0, eps: float = 1e-12) -> FloatArray:
    if lambda_a < 0 or lambda_b < 0:
        raise ValueError("lambda_a and lambda_b must be non-negative.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    q_b, _ = _prepare_batched(q, 2)
    m_b, _ = _prepare_batched(reference, 2)
    pa_b, _ = _prepare_batched(p_a, 1)
    pb_b, _ = _prepare_batched(p_b, 1)
    if q_b.shape != m_b.shape:
        raise ValueError("q and reference must have identical shapes.")
    n, k_a, k_b = q_b.shape
    if pa_b.shape != (n, k_a) or pb_b.shape != (n, k_b):
        raise ValueError("Marginal shapes are incompatible with joint shapes.")
    q_b = np.maximum(q_b, eps)
    q_b /= q_b.sum(axis=(1, 2), keepdims=True)
    m_b = np.maximum(m_b, eps)
    m_b /= m_b.sum(axis=(1, 2), keepdims=True)
    pa_b = _normalize_rows(pa_b, eps)
    pb_b = _normalize_rows(pb_b, eps)
    qa = q_b.sum(axis=2)
    qb = q_b.sum(axis=1)
    joint_term = _kl_per_sample(q_b, m_b, (1, 2), eps)
    a_term = _kl_per_sample(qa, pa_b, (1,), eps)
    b_term = _kl_per_sample(qb, pb_b, (1,), eps)
    return joint_term + lambda_a * a_term + lambda_b * b_term


def soft_reconciliation_eg(j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, *, reference_weight: float = 0.5, lambda_a: float = 1.0, lambda_b: float = 1.0, eps: float = 1e-12, tol: float = 1e-9, max_iter: int = 2_000, initial_step: float | None = None, max_backtracks: int = 25) -> SoftReconciliationResult:
    if not 0 <= reference_weight <= 1:
        raise ValueError("reference_weight must lie in [0, 1].")
    if lambda_a < 0 or lambda_b < 0:
        raise ValueError("lambda_a and lambda_b must be non-negative.")
    if eps <= 0 or tol <= 0 or max_iter < 1 or max_backtracks < 1:
        raise ValueError("Invalid optimization settings.")
    j1_b, squeeze = _prepare_batched(j1, 2)
    j2_b, squeeze2 = _prepare_batched(j2, 2)
    pa_b, squeeze3 = _prepare_batched(p_a, 1)
    pb_b, squeeze4 = _prepare_batched(p_b, 1)
    if not (squeeze == squeeze2 == squeeze3 == squeeze4):
        raise ValueError("Inputs must either all be single-sample or all batched.")
    if j1_b.shape != j2_b.shape:
        raise ValueError("j1 and j2 must have identical shapes.")
    n, k_a, k_b = j1_b.shape
    if pa_b.shape != (n, k_a) or pb_b.shape != (n, k_b):
        raise ValueError("Marginal shapes are incompatible with joint shapes.")
    pa_b = _normalize_rows(pa_b, eps)
    pb_b = _normalize_rows(pb_b, eps)
    reference = geometric_pool(j1_b, j2_b, weight=reference_weight, eps=eps)
    reference = np.maximum(reference, eps)
    reference /= reference.sum(axis=(1, 2), keepdims=True)
    q = reference.copy()
    obj = soft_reconciliation_objective(q, reference, pa_b, pb_b, lambda_a=lambda_a, lambda_b=lambda_b, eps=eps)
    initial_obj = obj.copy()
    if lambda_a == 0 and lambda_b == 0:
        conv = np.ones(n, dtype=bool)
        iters = np.zeros(n, dtype=np.int64)
        updates = np.zeros(n, dtype=np.float64)
        if squeeze:
            return SoftReconciliationResult(q[0], conv[:1], iters[:1], initial_obj[:1], obj[:1], updates[:1])
        return SoftReconciliationResult(q, conv, iters, initial_obj, obj, updates)
    base_step = initial_step if initial_step is not None else 1.0 / (1.0 + lambda_a + lambda_b)
    if base_step <= 0:
        raise ValueError("initial_step must be positive.")
    steps = np.full(n, float(base_step), dtype=np.float64)
    converged = np.zeros(n, dtype=bool)
    iterations = np.zeros(n, dtype=np.int64)
    max_update = np.full(n, np.inf, dtype=np.float64)
    log_m = np.log(np.maximum(reference, eps))
    log_pa = np.log(np.maximum(pa_b, eps))
    log_pb = np.log(np.maximum(pb_b, eps))
    for it in range(1, max_iter + 1):
        active = ~converged
        if not np.any(active):
            break
        qa = np.maximum(q.sum(axis=2), eps)
        qb = np.maximum(q.sum(axis=1), eps)
        grad = np.log(np.maximum(q, eps)) - log_m + lambda_a * (np.log(qa) - log_pa)[:, :, None] + lambda_b * (np.log(qb) - log_pb)[:, None, :]
        old_q = q.copy()
        old_obj = obj.copy()
        pending = active.copy()
        accepted = np.zeros(n, dtype=bool)
        for _ in range(max_backtracks):
            ids = np.where(pending)[0]
            if len(ids) == 0:
                break
            logits = np.log(np.maximum(old_q[ids], eps)) - steps[ids, None, None] * grad[ids]
            logits -= logits.max(axis=(1, 2), keepdims=True)
            proposal = np.exp(logits)
            proposal /= proposal.sum(axis=(1, 2), keepdims=True)
            proposal_obj = soft_reconciliation_objective(proposal, reference[ids], pa_b[ids], pb_b[ids], lambda_a=lambda_a, lambda_b=lambda_b, eps=eps)
            good = proposal_obj <= old_obj[ids] + 1e-14
            good_ids = ids[good]
            if len(good_ids):
                q[good_ids] = proposal[good]
                obj[good_ids] = proposal_obj[good]
                accepted[good_ids] = True
                pending[good_ids] = False
            bad_ids = ids[~good]
            if len(bad_ids):
                steps[bad_ids] *= 0.5
        stuck = active & ~accepted
        max_update[active] = np.sum(np.abs(q[active] - old_q[active]), axis=(1, 2))
        objective_change = np.abs(old_obj - obj)
        newly = active & ((max_update <= tol) | (objective_change <= tol * np.maximum(1.0, np.abs(old_obj))) | (stuck & (steps < 1e-14)))
        iterations[newly] = it
        converged |= newly
        successful = accepted & ~converged
        steps[successful] = np.minimum(steps[successful] * 1.05, base_step)
    iterations[~converged] = max_iter
    q /= q.sum(axis=(1, 2), keepdims=True)
    if squeeze:
        return SoftReconciliationResult(q[0], converged[:1], iterations[:1], initial_obj[:1], obj[:1], max_update[:1])
    return SoftReconciliationResult(q, converged, iterations, initial_obj, obj, max_update)


def soft_reconciliation(j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, *, reference_weight: float = 0.5, lambda_a: float = 1.0, lambda_b: float = 1.0, eps: float = 1e-12, tol: float = 1e-10, max_iter: int = 10_000) -> SoftReconciliationResult:
    if not 0 <= reference_weight <= 1:
        raise ValueError("reference_weight must lie in [0, 1].")
    if lambda_a < 0 or lambda_b < 0:
        raise ValueError("lambda_a and lambda_b must be non-negative.")
    if eps <= 0 or tol <= 0 or max_iter < 1:
        raise ValueError("eps/tol must be positive and max_iter >= 1.")
    j1_b, squeeze = _prepare_batched(j1, 2)
    j2_b, squeeze2 = _prepare_batched(j2, 2)
    pa_b, squeeze3 = _prepare_batched(p_a, 1)
    pb_b, squeeze4 = _prepare_batched(p_b, 1)
    if not (squeeze == squeeze2 == squeeze3 == squeeze4):
        raise ValueError("Inputs must either all be single-sample or all batched.")
    if j1_b.shape != j2_b.shape:
        raise ValueError("j1 and j2 must have identical shapes.")
    n, k_a, k_b = j1_b.shape
    if pa_b.shape != (n, k_a) or pb_b.shape != (n, k_b):
        raise ValueError("Marginal shapes are incompatible with joint shapes.")
    pa_b = _normalize_rows(pa_b, eps)
    pb_b = _normalize_rows(pb_b, eps)
    reference = geometric_pool(j1_b, j2_b, weight=reference_weight, eps=eps)
    reference = np.maximum(reference, eps)
    reference /= reference.sum(axis=(1, 2), keepdims=True)
    initial_obj = soft_reconciliation_objective(reference, reference, pa_b, pb_b, lambda_a=lambda_a, lambda_b=lambda_b, eps=eps)
    if lambda_a == 0 and lambda_b == 0:
        conv = np.ones(n, dtype=bool)
        iters = np.zeros(n, dtype=np.int64)
        updates = np.zeros(n, dtype=np.float64)
        if squeeze:
            return SoftReconciliationResult(reference[0], conv[:1], iters[:1], initial_obj[:1], initial_obj[:1], updates[:1])
        return SoftReconciliationResult(reference, conv, iters, initial_obj, initial_obj.copy(), updates)
    tau_a = lambda_a / (1.0 + lambda_a) if lambda_a > 0 else 0.0
    tau_b = lambda_b / (1.0 + lambda_b) if lambda_b > 0 else 0.0
    u = np.ones((n, k_a), dtype=np.float64)
    v = np.ones((n, k_b), dtype=np.float64)
    q = reference.copy()
    converged = np.zeros(n, dtype=bool)
    iterations = np.full(n, max_iter, dtype=np.int64)
    max_update = np.full(n, np.inf, dtype=np.float64)
    for it in range(1, max_iter + 1):
        old_q = q.copy()
        if lambda_a > 0:
            mv = np.einsum("nij,nj->ni", reference, v)
            u = np.power(pa_b / np.maximum(mv, eps), tau_a)
        else:
            u.fill(1.0)
        if lambda_b > 0:
            mtu = np.einsum("nij,ni->nj", reference, u)
            v = np.power(pb_b / np.maximum(mtu, eps), tau_b)
        else:
            v.fill(1.0)
        q = reference * u[:, :, None] * v[:, None, :]
        q /= np.maximum(q.sum(axis=(1, 2), keepdims=True), eps)
        max_update = np.max(np.abs(q - old_q), axis=(1, 2))
        newly = (~converged) & (max_update <= tol)
        iterations[newly] = it
        converged |= newly
        if np.all(converged):
            break
    final_obj = soft_reconciliation_objective(q, reference, pa_b, pb_b, lambda_a=lambda_a, lambda_b=lambda_b, eps=eps)
    if squeeze:
        return SoftReconciliationResult(q[0], converged[:1], iterations[:1], initial_obj[:1], final_obj[:1], max_update[:1])
    return SoftReconciliationResult(q, converged, iterations, initial_obj, final_obj, max_update)
