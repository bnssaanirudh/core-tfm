import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core_tfm.models.sklearn_like import SklearnLikeAdapter
from core_tfm.inference.extract import extract_pair_predictions


def test_extraction_shapes_with_sklearn_model():
    rng = np.random.default_rng(3)
    n = 180
    x = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("wxyz"))
    a = pd.Series((x["w"] + 0.4 * x["x"] > 0).astype(int), name="a")
    b = pd.Series((x["y"] + 0.8 * a + rng.normal(scale=0.5, size=n) > 0.4).astype(int), name="b")
    tr, te = np.arange(120), np.arange(120, n)

    def factory():
        return SklearnLikeAdapter(lambda: LogisticRegression(max_iter=500))

    out = extract_pair_predictions(factory, x.iloc[tr].reset_index(drop=True), a.iloc[tr].reset_index(drop=True), b.iloc[tr].reset_index(drop=True), x.iloc[te].reset_index(drop=True), a_test=a.iloc[te].reset_index(drop=True), b_test=b.iloc[te].reset_index(drop=True))
    pred = out.predictions
    assert pred.p_a.shape == (60, 2)
    assert pred.p_b.shape == (60, 2)
    assert pred.p_a_given_b.shape == (60, 2, 2)
    assert pred.p_b_given_a.shape == (60, 2, 2)
    np.testing.assert_allclose(pred.j_b_then_a.sum(axis=(1, 2)), 1.0, atol=1e-10)
    np.testing.assert_allclose(pred.j_a_then_b.sum(axis=(1, 2)), 1.0, atol=1e-10)
