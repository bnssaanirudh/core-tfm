import numpy as np
from core_tfm.reconciliation.weighted import select_geometric_pool_weight, select_reference_weight, select_soft_hyperparameters


def _case():
    n = 40
    p_a = np.tile([0.6, 0.4], (n, 1))
    p_b = np.tile([0.7, 0.3], (n, 1))
    j1 = np.tile([[0.55, 0.05], [0.15, 0.25]], (n, 1, 1))
    j2 = np.tile([[0.35, 0.25], [0.35, 0.05]], (n, 1, 1))
    ya = np.zeros(n, dtype=int)
    yb = np.zeros(n, dtype=int)
    return j1, j2, p_a, p_b, ya, yb


def test_weight_selector_prefers_better_direction_in_simple_case():
    j1, j2, p_a, p_b, ya, yb = _case()
    out = select_reference_weight(j1, j2, p_a, p_b, ya, yb, candidates=[0.0, 0.5, 1.0])
    assert out.weight == 1.0


def test_geometric_weight_selector_prefers_better_direction():
    j1, j2, _, _, ya, yb = _case()
    out = select_geometric_pool_weight(j1, j2, ya, yb, candidates=[0.0, 0.5, 1.0])
    assert out.weight == 1.0


def test_soft_grid_search_returns_a_candidate():
    j1, j2, p_a, p_b, ya, yb = _case()
    out = select_soft_hyperparameters(j1, j2, p_a, p_b, ya, yb, weights=[0.0, 0.5, 1.0], marginal_penalties=[0.1, 1.0])
    assert out.weight in {0.0, 0.5, 1.0}
    assert out.marginal_penalty in {0.1, 1.0}
    assert np.isfinite(out.score)
