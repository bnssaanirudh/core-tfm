from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .baselines import geometric_pool

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MPRResult:
    joint: FloatArray
    converged: NDArray[np.bool_]
    iterations: NDArray[np.int64]
    marginal_error: FloatArray


def _prepare_batched(x: ArrayLike, ndim_single: int) -> tuple[FloatArray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == ndim_single:
        return arr[None, ...], True
    if arr.ndim == ndim_single + 1:
        return arr, False
    raise ValueError(f"Expected {ndim_single}-D or {ndim_single + 1}-D array, got {arr.ndim}-D.")


def project_to_marginals(reference: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, *, eps: float = 1e-12, tol: float = 1e-10, max_iter: int = 10_000) -> MPRResult:
    """KL-project a positive reference joint onto prescribed marginals via IPF."""
    if eps <= 0:
        raise ValueError("eps must be positive for a full-support KL projection.")
    if tol <= 0 or max_iter < 1:
        raise ValueError("tol must be positive and max_iter >= 1.")
    q, squeeze_joint = _prepare_batched(reference, 2)
    p_a_b, squeeze_a = _prepare_batched(p_a, 1)
    p_b_b, squeeze_b = _prepare_batched(p_b, 1)
    if not (squeeze_joint == squeeze_a == squeeze_b):
        raise ValueError("Inputs must either all be single-sample or all batched.")
    n, k_a, k_b = q.shape
    if p_a_b.shape != (n, k_a) or p_b_b.shape != (n, k_b):
        raise ValueError("Marginal shapes are incompatible with joint shapes.")
    p_a_b = np.maximum(p_a_b, 0.0)
    p_b_b = np.maximum(p_b_b, 0.0)
    p_a_b /= p_a_b.sum(axis=1, keepdims=True)
    p_b_b /= p_b_b.sum(axis=1, keepdims=True)
    q = np.maximum(q, eps)
    q /= q.sum(axis=(1, 2), keepdims=True)
    converged = np.zeros(n, dtype=bool)
    iterations = np.zeros(n, dtype=np.int64)
    errors = np.full(n, np.inf, dtype=np.float64)
    for it in range(1, max_iter + 1):
        row_sum = q.sum(axis=2)
        row_scale = np.divide(p_a_b, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
        q *= row_scale[:, :, None]
        col_sum = q.sum(axis=1)
        col_scale = np.divide(p_b_b, col_sum, out=np.zeros_like(col_sum), where=col_sum > 0)
        q *= col_scale[:, None, :]
        row_err = np.abs(q.sum(axis=2) - p_a_b).sum(axis=1)
        col_err = np.abs(q.sum(axis=1) - p_b_b).sum(axis=1)
        errors = np.maximum(row_err, col_err)
        newly = (~converged) & (errors <= tol)
        iterations[newly] = it
        converged |= newly
        if np.all(converged):
            break
    iterations[~converged] = max_iter
    q /= q.sum(axis=(1, 2), keepdims=True)
    if squeeze_joint:
        return MPRResult(q[0], converged[:1], iterations[:1], errors[:1])
    return MPRResult(q, converged, iterations, errors)


def marginal_preserving_reconciliation(j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, *, reference_weight: ArrayLike = 0.5, eps: float = 1e-12, tol: float = 1e-10, max_iter: int = 10_000) -> MPRResult:
    reference = geometric_pool(j1, j2, weight=reference_weight, eps=eps)
    return project_to_marginals(reference, p_a, p_b, eps=eps, tol=tol, max_iter=max_iter)
