from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _normalize_joint(q: ArrayLike, eps: float = 0.0) -> FloatArray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 2:
        q = q[None, ...]
        squeeze = True
    elif q.ndim == 3:
        squeeze = False
    else:
        raise ValueError("Joint distributions must be 2-D or batched 3-D arrays.")
    if np.any(~np.isfinite(q)) or np.any(q < -1e-12):
        raise ValueError("Joint distributions must be finite and non-negative.")
    q = np.maximum(q, eps)
    z = q.sum(axis=(1, 2), keepdims=True)
    if np.any(z <= 0):
        raise ValueError("Joint distributions must have positive total mass.")
    q = q / z
    return q[0] if squeeze else q


def arithmetic_pool(j1: ArrayLike, j2: ArrayLike) -> FloatArray:
    j1 = _normalize_joint(j1)
    j2 = _normalize_joint(j2)
    if j1.shape != j2.shape:
        raise ValueError("j1 and j2 must have identical shapes.")
    return _normalize_joint(0.5 * (j1 + j2))


def geometric_pool(j1: ArrayLike, j2: ArrayLike, weight: ArrayLike = 0.5, eps: float = 1e-12) -> FloatArray:
    j1 = _normalize_joint(j1)
    j2 = _normalize_joint(j2)
    if j1.shape != j2.shape:
        raise ValueError("j1 and j2 must have identical shapes.")

    w = np.asarray(weight, dtype=np.float64)
    if np.any((w < 0.0) | (w > 1.0)):
        raise ValueError("weight must lie in [0, 1].")
    if w.ndim == 0:
        wj = w
    elif w.ndim == 1:
        if j1.ndim != 3 or w.shape[0] != j1.shape[0]:
            raise ValueError("Vector weights require batched joints and one weight per sample.")
        wj = w[:, None, None]
    else:
        raise ValueError("weight must be a scalar or a 1-D per-sample vector.")

    log_q = wj * np.log(np.maximum(j1, eps)) + (1.0 - wj) * np.log(np.maximum(j2, eps))
    log_q -= np.max(log_q, axis=(-2, -1), keepdims=True)
    return _normalize_joint(np.exp(log_q))


def independent_joint(p_a: ArrayLike, p_b: ArrayLike) -> FloatArray:
    p_a = np.asarray(p_a, dtype=np.float64)
    p_b = np.asarray(p_b, dtype=np.float64)
    if p_a.ndim == 1:
        p_a = p_a[None, ...]
        p_b = p_b[None, ...]
        squeeze = True
    else:
        squeeze = False
    p_a = p_a / p_a.sum(axis=1, keepdims=True)
    p_b = p_b / p_b.sum(axis=1, keepdims=True)
    q = p_a[:, :, None] * p_b[:, None, :]
    return q[0] if squeeze else q
