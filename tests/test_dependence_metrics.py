import numpy as np

from core_tfm.metrics.distributions import joint_mutual_information
from core_tfm.metrics.scoring import joint_expected_calibration_error


def test_mutual_information_zero_for_independence():
    pa = np.array([0.4, 0.6])
    pb = np.array([0.7, 0.3])
    q = pa[:, None] * pb[None, :]
    assert abs(float(joint_mutual_information(q))) < 1e-12


def test_mutual_information_positive_for_dependence():
    q = np.array([[0.48, 0.02], [0.02, 0.48]])
    assert float(joint_mutual_information(q)) > 0.4


def test_joint_ece_is_finite():
    q = np.array([[[0.7, 0.1], [0.1, 0.1]], [[0.1, 0.2], [0.2, 0.5]]])
    score = joint_expected_calibration_error(q, [0, 1], [0, 1], n_bins=5)
    assert 0.0 <= score <= 1.0
