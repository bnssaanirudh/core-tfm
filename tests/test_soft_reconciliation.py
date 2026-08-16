import numpy as np
from scipy.optimize import minimize

from core_tfm.metrics.distributions import marginal_distortion
from core_tfm.reconciliation.baselines import geometric_pool
from core_tfm.reconciliation.soft import soft_reconciliation, soft_reconciliation_objective


def _reference_slsqp(j1, j2, pa, pb, lam):
    m = geometric_pool(j1, j2)
    k_a, k_b = m.shape
    def unpack(z):
        z = z - np.max(z)
        q = np.exp(z).reshape(k_a, k_b)
        return q / q.sum()
    def fun(z):
        q = unpack(z)
        return float(soft_reconciliation_objective(q, m, pa, pb, lambda_a=lam, lambda_b=lam)[0])
    out = minimize(fun, np.log(m.ravel() + 1e-12), method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
    return unpack(out.x), out.fun


def test_soft_zero_penalty_is_geometric_pool():
    j1 = np.array([[0.42, 0.08], [0.18, 0.32]])
    j2 = np.array([[0.20, 0.30], [0.25, 0.25]])
    pa = np.array([0.55, 0.45])
    pb = np.array([0.60, 0.40])
    expected = geometric_pool(j1, j2)
    result = soft_reconciliation(j1, j2, pa, pb, lambda_a=0.0, lambda_b=0.0)
    np.testing.assert_allclose(result.joint, expected, atol=1e-12)
    assert result.converged[0]


def test_soft_fixed_point_when_reference_marginals_match():
    q = np.array([[0.30, 0.20], [0.10, 0.40]])
    pa, pb = q.sum(axis=1), q.sum(axis=0)
    result = soft_reconciliation(q, q, pa, pb, lambda_a=10.0, lambda_b=10.0)
    np.testing.assert_allclose(result.joint, q, atol=1e-8)


def test_larger_lambda_reduces_marginal_distortion():
    j1 = np.array([[0.65, 0.05], [0.05, 0.25]])
    j2 = np.array([[0.10, 0.35], [0.35, 0.20]])
    pa = np.array([0.8, 0.2])
    pb = np.array([0.25, 0.75])
    low = soft_reconciliation(j1, j2, pa, pb, lambda_a=0.1, lambda_b=0.1).joint
    high = soft_reconciliation(j1, j2, pa, pb, lambda_a=30.0, lambda_b=30.0).joint
    assert marginal_distortion(high, pa, pb)[0] < marginal_distortion(low, pa, pb)[0]


def test_soft_matches_small_generic_optimizer():
    j1 = np.array([[0.58, 0.07], [0.12, 0.23]])
    j2 = np.array([[0.18, 0.27], [0.31, 0.24]])
    pa = np.array([0.68, 0.32])
    pb = np.array([0.43, 0.57])
    lam = 2.0
    result = soft_reconciliation(j1, j2, pa, pb, lambda_a=lam, lambda_b=lam, tol=1e-11)
    _, ref_obj = _reference_slsqp(j1, j2, pa, pb, lam)
    m = geometric_pool(j1, j2)
    got_obj = soft_reconciliation_objective(result.joint, m, pa, pb, lambda_a=lam, lambda_b=lam)[0]
    assert got_obj <= ref_obj + 2e-7
