import numpy as np

from core_tfm.data.synthetic import make_multiclass_dgp
from core_tfm.data.view_perturbations import exact_views_from_joint, perturb_four_views
from core_tfm.metrics.distributions import total_variation


def test_exact_views_reconstruct_joint():
    dgp = make_multiclass_dgp(n=50, d=4, k_a=3, k_b=4, seed=5)
    v = exact_views_from_joint(dgp.true_joint)
    assert np.max(total_variation(v.j1_b_then_a, dgp.true_joint)) < 1e-10
    assert np.max(total_variation(v.j2_a_then_b, dgp.true_joint)) < 1e-10


def test_perturbations_create_valid_incompatible_views():
    dgp = make_multiclass_dgp(n=50, d=4, k_a=3, k_b=3, seed=6)
    v = perturb_four_views(dgp.true_joint, direct_noise=0.3, a_given_b_noise=0.4, b_given_a_noise=0.1, seed=7)
    assert np.allclose(v.p_a.sum(axis=1), 1)
    assert np.allclose(v.p_b.sum(axis=1), 1)
    assert np.allclose(v.j1_b_then_a.sum(axis=(1, 2)), 1)
    assert np.allclose(v.j2_a_then_b.sum(axis=(1, 2)), 1)
    assert float(total_variation(v.j1_b_then_a, v.j2_a_then_b).mean()) > 1e-4
