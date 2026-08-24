import numpy as np

from core_tfm.research_extensions import (
    audit_probability_views,
    complexity_penalty,
    decision_regret,
    inconsistency_accuracy_counterexamples,
    inverse_dispersion_weight,
    jensen_shannon_dispersion,
    oracle_selection_decomposition,
    safe_select_from_scores,
    support_adaptive_penalties,
)


def test_audit_zero_for_consistent_views():
    j = np.array([[[0.3, 0.2], [0.1, 0.4]]])
    pa = j.sum(axis=2)
    pb = j.sum(axis=1)
    audit = audit_probability_views(j, j, pa, pb)
    assert np.isclose(audit["mean_factorization_tv"], 0.0)
    assert np.isclose(audit["mean_marginalization_defect_a"], 0.0)
    assert np.isclose(audit["mean_marginalization_defect_b"], 0.0)


def test_js_dispersion_zero_for_identical_members():
    p = np.array([[0.2, 0.8], [0.7, 0.3]])
    stacked = np.stack([p, p, p], axis=0)
    assert np.allclose(jensen_shannon_dispersion(stacked), 0.0)


def test_inverse_dispersion_prefers_more_stable_view():
    w = inverse_dispersion_weight(np.array([0.01]), np.array([0.2]), temperature=0.1)
    assert w[0] > 0.5


def test_support_penalty_is_monotone():
    lam = support_adaptive_penalties([0, 1, 10, 100], base_lambda=10, tau=10)
    assert np.all(np.diff(lam) >= 0)
    assert lam[0] == 0
    assert lam[-1] < 10


def test_complexity_penalty_shrinks_with_validation_size():
    assert complexity_penalty(200, 48) < complexity_penalty(50, 48)


def test_safe_selector_can_fallback():
    families = {
        "simple": {"arithmetic": 1.0},
        "large": {"arithmetic": 1.0, "soft": 0.999, "mpr": 1.01, "j1": 1.02},
    }
    candidate, family, _ = safe_select_from_scores(families, n_validation=52, beta=1.0)
    assert candidate == "arithmetic"
    assert family == "fallback"


def test_oracle_decomposition_identity():
    d = oracle_selection_decomposition(
        {"arithmetic": 1.0, "selective": 1.1, "soft": 0.9},
        selected_method="selective",
    )
    assert np.isclose(d.opportunity, 0.1)
    assert np.isclose(d.selection_regret, 0.2)
    assert np.isclose(d.selected_vs_arithmetic, 0.1)


def test_decision_regret_zero_for_truth():
    p = np.array([[[0.4, 0.1], [0.2, 0.3]]])
    utility = np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])
    assert np.allclose(decision_regret(p, p, utility), 0.0)


def test_counterexamples_have_expected_direction():
    cx = inconsistency_accuracy_counterexamples([0.01, 0.05])
    assert all(r["arithmetic_regret"] > 0 for r in cx["large_tv_no_need_to_repair"])
    assert all(abs(r["arithmetic_regret"]) < 1e-12 for r in cx["small_tv_repair_can_help"])
