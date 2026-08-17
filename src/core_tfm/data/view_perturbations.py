from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _normalize(p: FloatArray, axis: int = -1, eps: float = 1e-12) -> FloatArray:
    p = np.maximum(np.asarray(p, dtype=np.float64), eps)
    return p / p.sum(axis=axis, keepdims=True)


def _softmax(z: FloatArray, axis: int = -1) -> FloatArray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def perturb_categorical(
    p: ArrayLike,
    *,
    temperature: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> FloatArray:
    """Temperature/noise perturbation of categorical distributions.

    The last axis is treated as the categorical support. ``temperature < 1``
    sharpens predictions, ``temperature > 1`` flattens them, and ``noise_std``
    adds independent Gaussian logit noise. The output remains strictly positive
    and normalized.
    """
    if temperature <= 0 or noise_std < 0:
        raise ValueError("temperature must be positive and noise_std non-negative")
    p = _normalize(np.asarray(p, dtype=np.float64), axis=-1, eps=eps)
    logits = np.log(np.maximum(p, eps)) / temperature
    if noise_std:
        logits = logits + rng.normal(scale=noise_std, size=logits.shape)
    return _softmax(logits, axis=-1)


@dataclass(frozen=True)
class FourViewPredictions:
    p_a: FloatArray
    p_b: FloatArray
    p_a_given_b: FloatArray  # (n, k_b, k_a)
    p_b_given_a: FloatArray  # (n, k_a, k_b)
    j1_b_then_a: FloatArray  # (n, k_a, k_b)
    j2_a_then_b: FloatArray  # (n, k_a, k_b)


def exact_views_from_joint(true_joint: ArrayLike, eps: float = 1e-12) -> FourViewPredictions:
    """Derive mutually compatible marginals/conditionals from a known joint."""
    q = np.asarray(true_joint, dtype=np.float64)
    if q.ndim != 3:
        raise ValueError("true_joint must have shape (n, k_a, k_b)")
    q = np.maximum(q, eps)
    q /= q.sum(axis=(1, 2), keepdims=True)
    pa = q.sum(axis=2)
    pb = q.sum(axis=1)
    # Conditional tensor orientations match inference.extract outputs.
    p_b_given_a = q / np.maximum(pa[:, :, None], eps)
    p_a_given_b = np.transpose(q / np.maximum(pb[:, None, :], eps), (0, 2, 1))
    return FourViewPredictions(pa, pb, p_a_given_b, p_b_given_a, q.copy(), q.copy())


def perturb_four_views(
    true_joint: ArrayLike,
    *,
    direct_temperature: float = 1.0,
    direct_noise: float = 0.0,
    a_given_b_temperature: float = 1.0,
    a_given_b_noise: float = 0.0,
    b_given_a_temperature: float = 1.0,
    b_given_a_noise: float = 0.0,
    seed: int = 0,
    eps: float = 1e-12,
) -> FourViewPredictions:
    """Generate deliberately incompatible four-view probabilistic predictions.

    This benchmark isolates *where* inconsistency enters. Direct marginals and
    the two conditional directions can be corrupted independently while the
    exact ground-truth joint remains known. It therefore complements surrogate
    model experiments, where errors in the four views are entangled.
    """
    exact = exact_views_from_joint(true_joint, eps=eps)
    rng = np.random.default_rng(seed)
    pa = perturb_categorical(
        exact.p_a,
        temperature=direct_temperature,
        noise_std=direct_noise,
        rng=rng,
        eps=eps,
    )
    pb = perturb_categorical(
        exact.p_b,
        temperature=direct_temperature,
        noise_std=direct_noise,
        rng=rng,
        eps=eps,
    )
    pab = perturb_categorical(
        exact.p_a_given_b,
        temperature=a_given_b_temperature,
        noise_std=a_given_b_noise,
        rng=rng,
        eps=eps,
    )
    pba = perturb_categorical(
        exact.p_b_given_a,
        temperature=b_given_a_temperature,
        noise_std=b_given_a_noise,
        rng=rng,
        eps=eps,
    )
    # pab is (n,k_b,k_a); transpose so A is the first joint axis.
    j1 = pa.dtype.type(1.0) * np.transpose(pab, (0, 2, 1)) * pb[:, None, :]
    j2 = pba * pa[:, :, None]
    j1 /= j1.sum(axis=(1, 2), keepdims=True)
    j2 /= j2.sum(axis=(1, 2), keepdims=True)
    return FourViewPredictions(pa, pb, pab, pba, j1, j2)
