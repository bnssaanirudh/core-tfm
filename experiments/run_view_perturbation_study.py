"""Known-truth four-view perturbation study.

Unlike the surrogate-model sweep, this experiment begins with an exact joint
P*(A,B|X) and corrupts direct marginals and each conditional direction
independently. It isolates when hard marginal preservation is appropriate,
when soft reconciliation is preferable, and when one autoregressive ordering
should simply be trusted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core_tfm.data.synthetic import make_multiclass_dgp
from core_tfm.data.view_perturbations import perturb_four_views
from core_tfm.metrics.distributions import total_variation, marginal_distortion, reconciliation_distortion
from core_tfm.metrics.scoring import joint_log_loss, joint_brier
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool, independent_joint
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.soft import soft_reconciliation
from core_tfm.reconciliation.weighted import select_soft_hyperparameters, select_original_order, select_geometric_pool_weight
from core_tfm.reconciliation.selective import select_reconciliation_policy, apply_reconciliation_policy


@dataclass(frozen=True)
class Regime:
    name: str
    direct_temperature: float = 1.0
    direct_noise: float = 0.0
    a_given_b_temperature: float = 1.0
    a_given_b_noise: float = 0.0
    b_given_a_temperature: float = 1.0
    b_given_a_noise: float = 0.0


REGIMES = [
    Regime("trusted_marginals", direct_noise=0.03, a_given_b_noise=0.28, b_given_a_noise=0.28),
    Regime("corrupted_marginals", direct_noise=0.42, a_given_b_noise=0.10, b_given_a_noise=0.10),
    Regime("asymmetric_j1_good", direct_noise=0.10, a_given_b_noise=0.05, b_given_a_noise=0.40),
    Regime("asymmetric_j2_good", direct_noise=0.10, a_given_b_noise=0.40, b_given_a_noise=0.05),
    Regime("overconfident_marginals", direct_temperature=0.55, direct_noise=0.08, a_given_b_noise=0.12, b_given_a_noise=0.12),
    Regime("underconfident_marginals", direct_temperature=1.80, direct_noise=0.08, a_given_b_noise=0.12, b_given_a_noise=0.12),
    Regime("symmetric_moderate", direct_noise=0.15, a_given_b_noise=0.15, b_given_a_noise=0.15),
]


def evaluate_seed(seed: int, regime: Regime, n: int, gamma: float):
    dgp = make_multiclass_dgp(n=n, d=12, k_a=3, k_b=3, gamma=gamma, nonlinear=True, seed=100_000 + seed)
    views = perturb_four_views(
        dgp.true_joint,
        direct_temperature=regime.direct_temperature,
        direct_noise=regime.direct_noise,
        a_given_b_temperature=regime.a_given_b_temperature,
        a_given_b_noise=regime.a_given_b_noise,
        b_given_a_temperature=regime.b_given_a_temperature,
        b_given_a_noise=regime.b_given_a_noise,
        seed=200_000 + seed,
    )
    idx = np.arange(n)
    va, te = train_test_split(idx, test_size=0.5, random_state=300_000 + seed, stratify=dgp.a)

    j1v, j2v = views.j1_b_then_a[va], views.j2_a_then_b[va]
    pav, pbv = views.p_a[va], views.p_b[va]
    yav, ybv = dgp.a[va], dgp.b[va]

    soft_sel = select_soft_hyperparameters(
        j1v, j2v, pav, pbv, yav, ybv,
        weights=(0.0, 0.25, 0.5, 0.75, 1.0),
        marginal_penalties=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    )
    pool_sel = select_geometric_pool_weight(j1v, j2v, yav, ybv)
    order_sel = select_original_order(j1v, j2v, yav, ybv)
    policy_sel = select_reconciliation_policy(
        j1v, j2v, pav, pbv, yav, ybv,
        weights=(0.25, 0.5, 0.75),
        marginal_penalties=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
    )

    j1, j2 = views.j1_b_then_a[te], views.j2_a_then_b[te]
    pa, pb = views.p_a[te], views.p_b[te]
    ya, yb = dgp.a[te], dgp.b[te]
    truth = dgp.true_joint[te]

    methods = {
        "j1": j1,
        "j2": j2,
        "selected_order": j1 if order_sel.name == "j1_b_then_a" else j2,
        "independent": independent_joint(pa, pb),
        "arithmetic": arithmetic_pool(j1, j2),
        "geometric": geometric_pool(j1, j2),
        "weighted_geometric": geometric_pool(j1, j2, weight=pool_sel.weight),
        "hard_core": marginal_preserving_reconciliation(j1, j2, pa, pb).joint,
        "adaptive_soft": soft_reconciliation(
            j1, j2, pa, pb,
            reference_weight=soft_sel.weight,
            lambda_a=soft_sel.marginal_penalty,
            lambda_b=soft_sel.marginal_penalty,
        ).joint,
        "selective_core": apply_reconciliation_policy(policy_sel, j1, j2, pa, pb),
    }
    best_order_nll = min(joint_log_loss(j1, ya, yb), joint_log_loss(j2, ya, yb))
    selected_order_nll = joint_log_loss(methods["selected_order"], ya, yb)
    rows = []
    factor_tv = float(total_variation(j1, j2).mean())
    for method, q in methods.items():
        nll = joint_log_loss(q, ya, yb)
        rows.append({
            "regime": regime.name,
            "seed": seed,
            "gamma": gamma,
            "method": method,
            "factorization_tv": factor_tv,
            "joint_nll": nll,
            "joint_brier": joint_brier(q, ya, yb),
            "tv_to_truth": float(total_variation(q, truth).mean()),
            "marginal_distortion": float(marginal_distortion(q, pa, pb).mean()),
            "reconciliation_distortion": float(reconciliation_distortion(q, j1, j2).mean()),
            "tax_vs_best_order": nll - best_order_nll,
            "tax_vs_selected_order": nll - selected_order_nll,
            "soft_weight": soft_sel.weight,
            "soft_lambda": soft_sel.marginal_penalty,
            "pool_weight": pool_sel.weight,
            "selected_order_name": order_sel.name,
            "selected_policy": policy_sel.policy.name,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--output", default="results/view_perturbation_study.csv")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    rows = []
    tasks = [(seed, regime, args.n, args.gamma) for regime in REGIMES for seed in range(args.seeds)]
    if args.jobs <= 1:
        for seed, regime, n, gamma in tasks:
            rows.extend(evaluate_seed(seed, regime, n, gamma))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(evaluate_seed, seed, regime, n, gamma): (seed, regime.name) for seed, regime, n, gamma in tasks}
            done = 0
            for fut in as_completed(futs):
                rows.extend(fut.result())
                done += 1
                if done % max(1, args.seeds) == 0:
                    print(f"completed {done}/{len(tasks)} seed-regime jobs")
    frame = pd.DataFrame(rows)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    summary = (
        frame.groupby(["regime", "method"], as_index=False)
        .agg(
            joint_nll_mean=("joint_nll", "mean"),
            joint_nll_sd=("joint_nll", "std"),
            tv_to_truth_mean=("tv_to_truth", "mean"),
            tax_vs_selected_mean=("tax_vs_selected_order", "mean"),
            marginal_distortion_mean=("marginal_distortion", "mean"),
        )
    )
    summary_path = path.with_name(path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved {path} and {summary_path}")


if __name__ == "__main__":
    main()
