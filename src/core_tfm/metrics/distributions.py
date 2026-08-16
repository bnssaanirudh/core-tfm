from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def total_variation(p: ArrayLike, q: ArrayLike, axis: tuple[int, ...] | int | None = None) -> FloatArray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have identical shapes.")
    if axis is None:
        axis = tuple(range(1, p.ndim)) if p.ndim > 1 else 0
    return 0.5 * np.sum(np.abs(p - q), axis=axis)


def _kl(p: FloatArray, q: FloatArray, axis, eps: float) -> FloatArray:
    p = np.maximum(p, 0.0)
    q = np.maximum(q, eps)
    terms = np.where(p > 0, p * (np.log(np.maximum(p, eps)) - np.log(q)), 0.0)
    return np.sum(terms, axis=axis)


def jensen_shannon(p: ArrayLike, q: ArrayLike, axis: tuple[int, ...] | int | None = None, eps: float = 1e-12) -> FloatArray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have identical shapes.")
    if axis is None:
        axis = tuple(range(1, p.ndim)) if p.ndim > 1 else 0
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, axis, eps) + 0.5 * _kl(q, m, axis, eps)


def reconciliation_distortion(q: ArrayLike, j1: ArrayLike, j2: ArrayLike) -> FloatArray:
    return 0.5 * (jensen_shannon(q, j1) + jensen_shannon(q, j2))


def marginal_distortion(q: ArrayLike, p_a: ArrayLike, p_b: ArrayLike) -> FloatArray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 2:
        q = q[None, ...]
    p_a = np.atleast_2d(np.asarray(p_a, dtype=np.float64))
    p_b = np.atleast_2d(np.asarray(p_b, dtype=np.float64))
    qa = q.sum(axis=2)
    qb = q.sum(axis=1)
    return 0.5 * (total_variation(qa, p_a, axis=1) + total_variation(qb, p_b, axis=1))


def joint_mutual_information(q: ArrayLike, eps: float = 1e-12) -> FloatArray:
    q = np.asarray(q, dtype=np.float64)
    squeeze = False
    if q.ndim == 2:
        q = q[None, ...]
        squeeze = True
    if q.ndim != 3:
        raise ValueError("q must have shape (k_a,k_b) or (n,k_a,k_b).")
    q = np.maximum(q, 0.0)
    q /= q.sum(axis=(1, 2), keepdims=True)
    qa = q.sum(axis=2, keepdims=True)
    qb = q.sum(axis=1, keepdims=True)
    indep = np.maximum(qa * qb, eps)
    terms = np.where(q > 0, q * (np.log(np.maximum(q, eps)) - np.log(indep)), 0.0)
    out = terms.sum(axis=(1, 2))
    return out[0] if squeeze else out
