import numpy as np

from core_tfm.reconciliation.selective import apply_reconciliation_policy, select_reconciliation_policy


def _toy():
    j1 = np.array([[[0.42, 0.08], [0.12, 0.38]], [[0.10, 0.40], [0.35, 0.15]], [[0.44, 0.06], [0.10, 0.40]], [[0.08, 0.42], [0.40, 0.10]]])
    j2 = np.array([[[0.36, 0.14], [0.18, 0.32]], [[0.14, 0.36], [0.31, 0.19]], [[0.38, 0.12], [0.16, 0.34]], [[0.13, 0.37], [0.34, 0.16]]])
    pa = np.array([[0.5, 0.5]] * 4)
    pb = np.array([[0.55, 0.45]] * 4)
    ya = np.array([0, 1, 0, 1])
    yb = np.array([0, 0, 0, 0])
    return j1, j2, pa, pb, ya, yb


def test_selective_policy_is_validation_only_and_applicable():
    j1, j2, pa, pb, ya, yb = _toy()
    sel = select_reconciliation_policy(j1, j2, pa, pb, ya, yb, weights=(0.25, 0.5, 0.75), marginal_penalties=(0.1, 1.0))
    q = apply_reconciliation_policy(sel, j1, j2, pa, pb)
    assert q.shape == j1.shape
    np.testing.assert_allclose(q.sum(axis=(1, 2)), 1.0, atol=1e-10)
    assert np.all(q >= 0)


def test_large_guard_can_force_original_order():
    j1, j2, pa, pb, ya, yb = _toy()
    sel = select_reconciliation_policy(j1, j2, pa, pb, ya, yb, marginal_penalties=(1.0,), min_improvement=100.0)
    assert sel.policy.name in {"j1_b_then_a", "j2_a_then_b"}
