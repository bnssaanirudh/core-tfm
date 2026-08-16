from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _sigmoid(z: FloatArray) -> FloatArray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class BinaryDGPData:
    x: FloatArray
    a: IntArray
    b: IntArray
    true_joint: FloatArray
    p_a: FloatArray
    p_b_given_a: FloatArray


def make_binary_dgp(n: int = 1000, d: int = 10, gamma: float = 1.5, *, nonlinear: bool = False, seed: int = 0) -> BinaryDGPData:
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive.")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    w_a = rng.normal(size=d) / np.sqrt(d)
    w_b = rng.normal(size=d) / np.sqrt(d)
    z_a = x @ w_a
    z_b_base = x @ w_b
    if nonlinear:
        if d >= 2:
            z_a += 0.7 * np.sin(x[:, 0] * x[:, 1])
        if d >= 3:
            z_b_base += 0.35 * (x[:, 2] ** 2 - 1.0)
        if d >= 5:
            z_b_base -= 0.5 * (x[:, 3] > 0) * x[:, 4]
    pa1 = _sigmoid(z_a)
    p_a = np.column_stack([1.0 - pa1, pa1])
    p_b_given_a = np.empty((n, 2, 2), dtype=np.float64)
    for a_val in (0, 1):
        pb1 = _sigmoid(z_b_base + gamma * (2 * a_val - 1))
        p_b_given_a[:, a_val, 0] = 1.0 - pb1
        p_b_given_a[:, a_val, 1] = pb1
    true_joint = p_a[:, :, None] * p_b_given_a
    a = np.array([rng.choice(2, p=p_a[i]) for i in range(n)], dtype=np.int64)
    b = np.array([rng.choice(2, p=p_b_given_a[i, a[i]]) for i in range(n)], dtype=np.int64)
    return BinaryDGPData(x=x, a=a, b=b, true_joint=true_joint, p_a=p_a, p_b_given_a=p_b_given_a)


def _softmax(z: FloatArray) -> FloatArray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class MulticlassDGPData:
    x: FloatArray
    a: IntArray
    b: IntArray
    true_joint: FloatArray
    p_a: FloatArray
    p_b_given_a: FloatArray


def make_multiclass_dgp(n: int = 1000, d: int = 10, k_a: int = 3, k_b: int = 3, gamma: float = 1.5, *, nonlinear: bool = False, class_bias: float = 0.0, seed: int = 0) -> MulticlassDGPData:
    if n < 1 or d < 1 or k_a < 2 or k_b < 2:
        raise ValueError("n/d must be positive and k_a/k_b must be >= 2.")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    w_a = rng.normal(size=(d, k_a)) / np.sqrt(d)
    w_b = rng.normal(size=(d, k_b)) / np.sqrt(d)
    logits_a = x @ w_a
    logits_b = x @ w_b
    if nonlinear:
        if d >= 2:
            term = np.sin(x[:, 0] * x[:, 1])
            coeff = np.linspace(-0.7, 0.7, k_a)
            logits_a += term[:, None] * coeff[None, :]
        if d >= 3:
            term = x[:, 2] ** 2 - 1.0
            coeff = np.linspace(0.4, -0.4, k_b)
            logits_b += term[:, None] * coeff[None, :]
        if d >= 5:
            term = (x[:, 3] > 0) * x[:, 4]
            coeff = np.linspace(-0.5, 0.5, k_b)
            logits_b += term[:, None] * coeff[None, :]
    if class_bias:
        logits_a[:, 0] += class_bias
        logits_b[:, 0] += class_bias
    p_a = _softmax(logits_a)
    effects = rng.normal(size=(k_a, k_b))
    effects -= effects.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(effects, axis=1, keepdims=True)
    effects /= np.where(norms > 0, norms, 1.0)
    p_b_given_a = np.empty((n, k_a, k_b), dtype=np.float64)
    for a_val in range(k_a):
        p_b_given_a[:, a_val, :] = _softmax(logits_b + gamma * effects[a_val])
    true_joint = p_a[:, :, None] * p_b_given_a
    a = np.array([rng.choice(k_a, p=p_a[i]) for i in range(n)], dtype=np.int64)
    b = np.array([rng.choice(k_b, p=p_b_given_a[i, a[i]]) for i in range(n)], dtype=np.int64)
    return MulticlassDGPData(x=x, a=a, b=b, true_joint=true_joint, p_a=p_a, p_b_given_a=p_b_given_a)
