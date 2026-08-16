from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

from core_tfm.metrics.distributions import total_variation
from .baselines import geometric_pool
from .mpr import project_to_marginals, MPRResult

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DirectionalDefects:
    j1_defect: FloatArray
    j2_defect: FloatArray

    @property
    def trust_gap(self) -> FloatArray:
        return self.j2_defect - self.j1_defect


def directional_marginal_defects(j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike) -> DirectionalDefects:
    j1 = np.asarray(j1, dtype=np.float64)
    j2 = np.asarray(j2, dtype=np.float64)
    pa = np.asarray(p_a, dtype=np.float64)
    pb = np.asarray(p_b, dtype=np.float64)
    if j1.ndim == 2:
        j1 = j1[None, ...]
        j2 = j2[None, ...]
        pa = np.atleast_2d(pa)
        pb = np.atleast_2d(pb)
    if j1.shape != j2.shape or j1.ndim != 3:
        raise ValueError("j1 and j2 must have identical (n, k_a, k_b) shapes.")
    if pa.shape != (j1.shape[0], j1.shape[1]):
        raise ValueError("p_a shape is incompatible with the joints.")
    if pb.shape != (j1.shape[0], j1.shape[2]):
        raise ValueError("p_b shape is incompatible with the joints.")
    d1 = total_variation(j1.sum(axis=2), pa, axis=1)
    d2 = total_variation(j2.sum(axis=1), pb, axis=1)
    return DirectionalDefects(j1_defect=d1, j2_defect=d2)


def inverse_defect_weights(defects: DirectionalDefects, *, smoothing: float = 1e-3) -> FloatArray:
    if smoothing <= 0:
        raise ValueError("smoothing must be positive.")
    d1 = np.asarray(defects.j1_defect, dtype=np.float64)
    d2 = np.asarray(defects.j2_defect, dtype=np.float64)
    return (d2 + smoothing) / (d1 + d2 + 2.0 * smoothing)


def softmax_defect_weights(defects: DirectionalDefects, *, temperature: float = 10.0) -> FloatArray:
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    d1 = np.asarray(defects.j1_defect, dtype=np.float64)
    d2 = np.asarray(defects.j2_defect, dtype=np.float64)
    z = np.clip(temperature * (d2 - d1), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def defect_weighted_pool(j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, *, smoothing: float = 1e-3, eps: float = 1e-12):
    defects = directional_marginal_defects(j1, j2, p_a, p_b)
    weights = inverse_defect_weights(defects, smoothing=smoothing)
    q = geometric_pool(j1, j2, weight=weights, eps=eps)
    return q, weights, defects


def defect_weighted_mpr(j1: ArrayLike, j2: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, *, smoothing: float = 1e-3, eps: float = 1e-12, tol: float = 1e-10, max_iter: int = 10_000):
    reference, weights, defects = defect_weighted_pool(j1, j2, p_a, p_b, smoothing=smoothing, eps=eps)
    result = project_to_marginals(reference, p_a, p_b, eps=eps, tol=tol, max_iter=max_iter)
    return result, weights, defects
