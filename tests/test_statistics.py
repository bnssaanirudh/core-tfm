import numpy as np
from core_tfm.metrics.statistics import holm_adjust, paired_bootstrap_mean_difference, paired_wilcoxon


def test_holm_adjust_is_bounded_and_monotone_in_sorted_order():
    p = np.array([0.01, 0.04, 0.03])
    adj = holm_adjust(p)
    assert np.all((adj >= p) & (adj <= 1))
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_bootstrap_difference_contains_obvious_shift():
    x = np.arange(20, dtype=float)
    y = x + 2.0
    mean, lo, hi = paired_bootstrap_mean_difference(x, y, n_boot=1000, seed=1)
    assert abs(mean + 2.0) < 1e-12
    assert lo <= -2.0 <= hi


def test_wilcoxon_identical_returns_one():
    _, p = paired_wilcoxon([1, 2, 3], [1, 2, 3])
    assert p == 1.0
