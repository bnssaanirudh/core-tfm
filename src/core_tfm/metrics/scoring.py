from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _check_probabilities(p: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if p.ndim != 2:
        raise ValueError("p must have shape (n, k).")
    if len(y) != p.shape[0]:
        raise ValueError("Labels and probabilities must have the same sample count.")
    if np.any(~np.isfinite(p)) or np.any(p < -1e-12):
        raise ValueError("Probabilities must be finite and non-negative.")
    p = np.maximum(p, 0.0)
    p /= p.sum(axis=1, keepdims=True)
    return p, y


def joint_log_loss(q: ArrayLike, y_a: ArrayLike, y_b: ArrayLike, eps: float = 1e-15) -> float:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 3:
        raise ValueError("q must have shape (n, k_a, k_b).")
    y_a = np.asarray(y_a, dtype=int)
    y_b = np.asarray(y_b, dtype=int)
    if len(y_a) != q.shape[0] or len(y_b) != q.shape[0]:
        raise ValueError("Labels and q must have the same sample count.")
    probs = q[np.arange(q.shape[0]), y_a, y_b]
    return float(-np.mean(np.log(np.maximum(probs, eps))))


def joint_brier(q: ArrayLike, y_a: ArrayLike, y_b: ArrayLike) -> float:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 3:
        raise ValueError("q must have shape (n, k_a, k_b).")
    y_a = np.asarray(y_a, dtype=int)
    y_b = np.asarray(y_b, dtype=int)
    k_b = q.shape[2]
    y_joint = y_a * k_b + y_b
    return multiclass_brier(q.reshape(q.shape[0], -1), y_joint)


def multiclass_log_loss(p: ArrayLike, y: ArrayLike, eps: float = 1e-15) -> float:
    p, y = _check_probabilities(p, y)
    probs = p[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.maximum(probs, eps))))


def accuracy_from_proba(p: ArrayLike, y: ArrayLike) -> float:
    p, y = _check_probabilities(p, y)
    return float(np.mean(p.argmax(axis=1) == y))


def multiclass_brier(p: ArrayLike, y: ArrayLike) -> float:
    p, y = _check_probabilities(p, y)
    target = np.zeros_like(p)
    target[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - target) ** 2, axis=1)))


def expected_calibration_error(p: ArrayLike, y: ArrayLike, n_bins: int = 15) -> float:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    p, y = _check_probabilities(p, y)
    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    correct = (prediction == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < n_bins - 1 else confidence <= hi)
        if np.any(mask):
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def joint_expected_calibration_error(q: ArrayLike, y_a: ArrayLike, y_b: ArrayLike, n_bins: int = 15) -> float:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 3:
        raise ValueError("q must have shape (n, k_a, k_b).")
    y_a = np.asarray(y_a, dtype=int)
    y_b = np.asarray(y_b, dtype=int)
    k_b = q.shape[2]
    y_joint = y_a * k_b + y_b
    return expected_calibration_error(q.reshape(q.shape[0], -1), y_joint, n_bins=n_bins)


def conditionals_from_joint(q: ArrayLike, eps: float = 1e-15) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 3:
        raise ValueError("q must have shape (n, k_a, k_b).")
    if np.any(~np.isfinite(q)) or np.any(q < -1e-12):
        raise ValueError("Joint probabilities must be finite and non-negative.")
    q = np.maximum(q, 0.0)
    q /= np.maximum(q.sum(axis=(1, 2), keepdims=True), eps)
    p_b = q.sum(axis=1, keepdims=True)
    p_a = q.sum(axis=2, keepdims=True)
    a_given_b = q / np.maximum(p_b, eps)
    b_given_a = q / np.maximum(p_a, eps)
    a_given_b /= np.maximum(a_given_b.sum(axis=1, keepdims=True), eps)
    b_given_a /= np.maximum(b_given_a.sum(axis=2, keepdims=True), eps)
    return a_given_b, b_given_a


def conditional_log_losses(q: ArrayLike, y_a: ArrayLike, y_b: ArrayLike, eps: float = 1e-15) -> tuple[float, float]:
    q = np.asarray(q, dtype=np.float64)
    y_a = np.asarray(y_a, dtype=int)
    y_b = np.asarray(y_b, dtype=int)
    if q.ndim != 3 or len(y_a) != q.shape[0] or len(y_b) != q.shape[0]:
        raise ValueError("Shapes of q and labels are incompatible.")
    a_given_b, b_given_a = conditionals_from_joint(q, eps=eps)
    rows = np.arange(q.shape[0])
    pa = a_given_b[rows, y_a, y_b]
    pb = b_given_a[rows, y_a, y_b]
    return float(-np.mean(np.log(np.maximum(pa, eps)))), float(-np.mean(np.log(np.maximum(pb, eps))))


def conditional_brier_scores(q: ArrayLike, y_a: ArrayLike, y_b: ArrayLike) -> tuple[float, float]:
    q = np.asarray(q, dtype=np.float64)
    y_a = np.asarray(y_a, dtype=int)
    y_b = np.asarray(y_b, dtype=int)
    if q.ndim != 3 or len(y_a) != q.shape[0] or len(y_b) != q.shape[0]:
        raise ValueError("Shapes of q and labels are incompatible.")
    a_given_b, b_given_a = conditionals_from_joint(q)
    rows = np.arange(q.shape[0])
    pa = a_given_b[rows, :, y_b]
    pb = b_given_a[rows, y_a, :]
    return multiclass_brier(pa, y_a), multiclass_brier(pb, y_b)


def conditional_expected_calibration_errors(q: ArrayLike, y_a: ArrayLike, y_b: ArrayLike, n_bins: int = 15) -> tuple[float, float]:
    q = np.asarray(q, dtype=np.float64)
    y_a = np.asarray(y_a, dtype=int)
    y_b = np.asarray(y_b, dtype=int)
    if q.ndim != 3 or len(y_a) != q.shape[0] or len(y_b) != q.shape[0]:
        raise ValueError("Shapes of q and labels are incompatible.")
    a_given_b, b_given_a = conditionals_from_joint(q)
    rows = np.arange(q.shape[0])
    pa = a_given_b[rows, :, y_b]
    pb = b_given_a[rows, y_a, :]
    return expected_calibration_error(pa, y_a, n_bins=n_bins), expected_calibration_error(pb, y_b, n_bins=n_bins)
