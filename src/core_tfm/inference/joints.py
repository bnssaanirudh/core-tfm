from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_probabilities(x: ArrayLike, axis: int = -1, eps: float = 0.0) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Probabilities contain non-finite values.")
    if np.any(arr < -1e-12):
        raise ValueError("Probabilities must be non-negative.")
    arr = np.maximum(arr, eps)
    denom = arr.sum(axis=axis, keepdims=True)
    if np.any(denom <= 0):
        raise ValueError("Each probability vector must have positive mass.")
    return arr / denom


@dataclass(frozen=True)
class JointPredictions:
    p_a: FloatArray
    p_b: FloatArray
    p_a_given_b: FloatArray
    p_b_given_a: FloatArray
    j_b_then_a: FloatArray
    j_a_then_b: FloatArray


def construct_implied_joints(p_a: ArrayLike, p_b: ArrayLike, p_a_given_b: ArrayLike, p_b_given_a: ArrayLike) -> JointPredictions:
    p_a = _as_probabilities(p_a)
    p_b = _as_probabilities(p_b)
    p_a_given_b = _as_probabilities(p_a_given_b)
    p_b_given_a = _as_probabilities(p_b_given_a)
    if p_a.ndim != 2 or p_b.ndim != 2:
        raise ValueError("p_a and p_b must be 2-D arrays.")
    n, k_a = p_a.shape
    n_b, k_b = p_b.shape
    if n != n_b:
        raise ValueError("p_a and p_b must have the same number of samples.")
    if p_a_given_b.shape != (n, k_b, k_a):
        raise ValueError(f"p_a_given_b must have shape {(n, k_b, k_a)}, got {p_a_given_b.shape}.")
    if p_b_given_a.shape != (n, k_a, k_b):
        raise ValueError(f"p_b_given_a must have shape {(n, k_a, k_b)}, got {p_b_given_a.shape}.")
    j_b_then_a = np.transpose(p_a_given_b, (0, 2, 1)) * p_b[:, None, :]
    j_a_then_b = p_a[:, :, None] * p_b_given_a
    j_b_then_a /= j_b_then_a.sum(axis=(1, 2), keepdims=True)
    j_a_then_b /= j_a_then_b.sum(axis=(1, 2), keepdims=True)
    return JointPredictions(p_a=p_a, p_b=p_b, p_a_given_b=p_a_given_b, p_b_given_a=p_b_given_a, j_b_then_a=j_b_then_a, j_a_then_b=j_a_then_b)
