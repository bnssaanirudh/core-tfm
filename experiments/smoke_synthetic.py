"""End-to-end smoke experiment without any TFM checkpoint.

The script deliberately uses logistic regression. Its purpose is to test the
research pipeline (conditional extraction -> incompatible joints -> repair ->
known-truth scoring), not to make claims about TFMs.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from core_tfm.data.synthetic import make_binary_dgp
from core_tfm.inference.extract import extract_pair_predictions
from core_tfm.models.sklearn_like import SklearnLikeAdapter
from core_tfm.metrics.distributions import marginal_distortion, total_variation
from core_tfm.metrics.scoring import joint_log_loss
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool, independent_joint
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.soft import soft_reconciliation
from core_tfm.reconciliation.weighted import select_reference_weight


def factory():
    return SklearnLikeAdapter(lambda: LogisticRegression(max_iter=1000))


def main():
    dgp = make_binary_dgp(n=1600, d=8, gamma=1.5, nonlinear=True, seed=7)
    X = pd.DataFrame(dgp.x, columns=[f"x{i}" for i in range(dgp.x.shape[1])])
    a = pd.Series(dgp.a, name="a")
    b = pd.Series(dgp.b, name="b")
    idx = np.arange(len(X))
    train_all, te = train_test_split(idx, test_size=0.3, random_state=42, stratify=a)
    tr, va = train_test_split(train_all, test_size=0.2, random_state=43, stratify=a.iloc[train_all])
    val = extract_pair_predictions(factory, X.iloc[tr].reset_index(drop=True), a.iloc[tr].reset_index(drop=True), b.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True), a_test=a.iloc[va].reset_index(drop=True), b_test=b.iloc[va].reset_index(drop=True))
    vp = val.predictions
    selected = select_reference_weight(vp.j_b_then_a, vp.j_a_then_b, vp.p_a, vp.p_b, val.y_a_encoded, val.y_b_encoded)
    out = extract_pair_predictions(factory, X.iloc[train_all].reset_index(drop=True), a.iloc[train_all].reset_index(drop=True), b.iloc[train_all].reset_index(drop=True), X.iloc[te].reset_index(drop=True), a_test=a.iloc[te].reset_index(drop=True), b_test=b.iloc[te].reset_index(drop=True))
    p = out.predictions
    j1, j2 = p.j_b_then_a, p.j_a_then_b
    qa = arithmetic_pool(j1, j2)
    qg = geometric_pool(j1, j2)
    qi = independent_joint(p.p_a, p.p_b)
    qm = marginal_preserving_reconciliation(j1, j2, p.p_a, p.p_b).joint
    qrw = marginal_preserving_reconciliation(j1, j2, p.p_a, p.p_b, reference_weight=selected.weight).joint
    qs = soft_reconciliation(j1, j2, p.p_a, p.p_b, lambda_a=1.0, lambda_b=1.0).joint
    truth = dgp.true_joint[te]
    rows = {"J1": j1, "J2": j2, "Independent": qi, "Arithmetic": qa, "Geometric": qg, "MPR": qm, f"RWR-MPR(w={selected.weight:g})": qrw, "Soft(lambda=1)": qs}
    print(f"Selected RWR validation weight: {selected.weight:g}; scores={selected.scores}")
    print(f"Mean original factorization TV: {total_variation(j1, j2).mean():.6f}")
    print("method              joint_nll   TV_to_true   marginal_distortion")
    for name, q in rows.items():
        nll = joint_log_loss(q, out.y_a_encoded, out.y_b_encoded)
        tv = total_variation(q, truth).mean()
        md = marginal_distortion(q, p.p_a, p.p_b).mean()
        print(f"{name:20s}  {nll:9.5f}   {tv:10.5f}   {md:19.6f}")


if __name__ == "__main__":
    main()
