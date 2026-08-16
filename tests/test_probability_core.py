import numpy as np

from core_tfm.inference.joints import construct_implied_joints
from core_tfm.metrics.distributions import total_variation


def test_construct_joints_for_coherent_binary_distribution():
    q = np.array([[0.30, 0.20], [0.10, 0.40]])
    p_a = q.sum(axis=1)[None, :]
    p_b = q.sum(axis=0)[None, :]
    p_a_given_b = np.stack([q[:, b] / p_b[0, b] for b in range(2)], axis=0)[None, :, :]
    p_b_given_a = np.stack([q[a, :] / p_a[0, a] for a in range(2)], axis=0)[None, :, :]
    out = construct_implied_joints(p_a, p_b, p_a_given_b, p_b_given_a)
    np.testing.assert_allclose(out.j_b_then_a[0], q, atol=1e-12)
    np.testing.assert_allclose(out.j_a_then_b[0], q, atol=1e-12)
    assert total_variation(out.j_b_then_a, out.j_a_then_b)[0] < 1e-12
