import numpy as np
from core_tfm.data.synthetic import make_binary_dgp, make_multiclass_dgp


def test_binary_dgp_joint_is_normalized_and_matches_pa():
    data = make_binary_dgp(n=100, d=5, gamma=1.0, seed=42)
    np.testing.assert_allclose(data.true_joint.sum(axis=(1, 2)), 1.0, atol=1e-12)
    np.testing.assert_allclose(data.true_joint.sum(axis=2), data.p_a, atol=1e-12)
    assert set(np.unique(data.a)).issubset({0, 1})
    assert set(np.unique(data.b)).issubset({0, 1})


def test_multiclass_dgp_joint_is_normalized_and_matches_pa():
    data = make_multiclass_dgp(n=120, d=7, k_a=3, k_b=5, gamma=1.2, nonlinear=True, seed=9)
    assert data.true_joint.shape == (120, 3, 5)
    np.testing.assert_allclose(data.true_joint.sum(axis=(1, 2)), 1.0, atol=1e-12)
    np.testing.assert_allclose(data.true_joint.sum(axis=2), data.p_a, atol=1e-12)
