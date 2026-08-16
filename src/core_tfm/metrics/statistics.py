from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import wilcoxon

FloatArray = NDArray[np.float64]


def holm_adjust(p_values: ArrayLike) -> FloatArray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a 1-D array in [0, 1].")
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted_sorted[rank] = running
    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted_sorted
    return out


def paired_bootstrap_mean_difference(x: ArrayLike, y: ArrayLike, *, n_boot: int = 10_000, confidence: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("x and y must be equally sized 1-D arrays with >=2 values.")
    if n_boot < 100 or not 0 < confidence < 1:
        raise ValueError("n_boot must be >=100 and confidence in (0,1).")
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def paired_wilcoxon(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be equally sized 1-D arrays.")
    if np.allclose(x, y):
        return 0.0, 1.0
    result = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)
