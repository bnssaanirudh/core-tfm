import numpy as np

from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool, independent_joint
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.metrics.distributions import marginal_distortion


def test_mpr_preserves_marginals():
    j1 = np.array([[0.42, 0.08], [0.18, 0.32]])
    j2 = np.array([[0.20, 0.30], [0.25, 0.25]])
    p_a = np.array([0.55, 0.45])
    p_b = np.array([0.60, 0.40])
    result = marginal_preserving_reconciliation(j1, j2, p_a, p_b)
    assert result.converged[0]
    np.testing.assert_allclose(result.joint.sum(axis=1), p_a, atol=1e-9)
    np.testing.assert_allclose(result.joint.sum(axis=0), p_b, atol=1e-9)
    assert marginal_distortion(result.joint, p_a, p_b)[0] < 1e-9


def test_mpr_fixed_point_when_consensus_already_has_required_marginals():
    q = np.array([[0.30, 0.20], [0.10, 0.40]])
    p_a = q.sum(axis=1)
    p_b = q.sum(axis=0)
    result = marginal_preserving_reconciliation(q, q, p_a, p_b)
    np.testing.assert_allclose(result.joint, q, atol=1e-9)


def test_baselines_are_normalized():
    j1 = np.array([[0.7, 0.1], [0.1, 0.1]])
    j2 = np.array([[0.1, 0.2], [0.3, 0.4]])
    for q in (arithmetic_pool(j1, j2), geometric_pool(j1, j2)):
        np.testing.assert_allclose(q.sum(), 1.0, atol=1e-12)
        assert np.all(q >= 0)
    qind = independent_joint(np.array([0.4, 0.6]), np.array([0.7, 0.3]))
    np.testing.assert_allclose(qind.sum(axis=1), [0.4, 0.6], atol=1e-12)
    np.testing.assert_allclose(qind.sum(axis=0), [0.7, 0.3], atol=1e-12)
