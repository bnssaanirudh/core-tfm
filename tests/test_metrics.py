import numpy as np
from core_tfm.metrics.distributions import total_variation, jensen_shannon, reconciliation_distortion
from core_tfm.metrics.scoring import accuracy_from_proba, expected_calibration_error, joint_brier, joint_log_loss, multiclass_brier, multiclass_log_loss


def test_metrics_basic_identities():
    p = np.array([[[0.3, 0.2], [0.1, 0.4]]])
    q = p.copy()
    np.testing.assert_allclose(total_variation(p, q), 0.0)
    np.testing.assert_allclose(jensen_shannon(p, q), 0.0)
    np.testing.assert_allclose(reconciliation_distortion(p, p, q), 0.0)
    expected = -np.log(0.4)
    assert abs(joint_log_loss(p, [1], [1]) - expected) < 1e-12


def test_classification_scoring_metrics():
    p = np.array([[0.8, 0.2], [0.3, 0.7]])
    y = np.array([0, 1])
    assert accuracy_from_proba(p, y) == 1.0
    assert multiclass_log_loss(p, y) > 0
    assert multiclass_brier(p, y) > 0
    assert 0 <= expected_calibration_error(p, y, n_bins=4) <= 1
    q = np.array([[[0.7, 0.1], [0.1, 0.1]], [[0.1, 0.2], [0.1, 0.6]]])
    assert joint_brier(q, [0, 1], [0, 1]) > 0


def test_conditionals_from_joint_and_conditional_scores():
    from core_tfm.metrics.scoring import conditionals_from_joint, conditional_log_losses, conditional_brier_scores
    q = np.array([[[0.4, 0.1], [0.2, 0.3]], [[0.1, 0.2], [0.3, 0.4]]])
    ab, ba = conditionals_from_joint(q)
    np.testing.assert_allclose(ab.sum(axis=1), 1.0)
    np.testing.assert_allclose(ba.sum(axis=2), 1.0)
    ca, cb = conditional_log_losses(q, np.array([0, 1]), np.array([0, 1]))
    ba_score, bb_score = conditional_brier_scores(q, np.array([0, 1]), np.array([0, 1]))
    assert np.isfinite(ca) and np.isfinite(cb)
    assert ba_score >= 0 and bb_score >= 0


def test_conditional_ece_is_zero_for_perfect_deterministic_joint():
    from core_tfm.metrics.scoring import conditional_expected_calibration_errors
    q = np.zeros((4, 2, 2), dtype=float)
    ya = np.array([0, 0, 1, 1])
    yb = np.array([0, 1, 0, 1])
    q[np.arange(4), ya, yb] = 1.0
    ea, eb = conditional_expected_calibration_errors(q, ya, yb, n_bins=5)
    assert ea == 0.0
    assert eb == 0.0
