import numpy as np

from core_tfm.reconciliation.defect import (
    directional_marginal_defects,
    inverse_defect_weights,
    defect_weighted_pool,
    defect_weighted_mpr,
)


def test_directional_defects_prefer_more_self_consistent_factorization():
    p_a = np.array([[0.6, 0.4]])
    p_b = np.array([[0.7, 0.3]])
    j1 = np.array([[[0.50, 0.08], [0.20, 0.22]]])
    j2 = np.array([[[0.20, 0.40], [0.10, 0.30]]])
    d = directional_marginal_defects(j1, j2, p_a, p_b)
    assert d.j1_defect[0] < d.j2_defect[0]
    w = inverse_defect_weights(d, smoothing=1e-3)
    assert w[0] > 0.5


def test_defect_weighted_pool_is_normalized_and_returns_weights():
    p_a = np.array([[0.6, 0.4], [0.5, 0.5]])
    p_b = np.array([[0.7, 0.3], [0.4, 0.6]])
    j1 = np.array([[[0.50, 0.08], [0.20, 0.22]], [[0.30, 0.20], [0.10, 0.40]]])
    j2 = np.array([[[0.20, 0.40], [0.10, 0.30]], [[0.25, 0.25], [0.15, 0.35]]])
    q, w, _ = defect_weighted_pool(j1, j2, p_a, p_b)
    np.testing.assert_allclose(q.sum(axis=(1, 2)), 1.0, atol=1e-12)
    assert np.all((w >= 0) & (w <= 1))


def test_defect_weighted_mpr_preserves_direct_marginals():
    p_a = np.array([[0.6, 0.4]])
    p_b = np.array([[0.7, 0.3]])
    j1 = np.array([[[0.50, 0.08], [0.20, 0.22]]])
    j2 = np.array([[[0.20, 0.40], [0.10, 0.30]]])
    result, weights, _ = defect_weighted_mpr(j1, j2, p_a, p_b)
    assert result.converged[0]
    assert weights.shape == (1,)
    np.testing.assert_allclose(result.joint.sum(axis=2), p_a, atol=1e-9)
    np.testing.assert_allclose(result.joint.sum(axis=1), p_b, atol=1e-9)
