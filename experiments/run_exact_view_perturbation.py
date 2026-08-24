"""Exact-view perturbation benchmark for CoRe-TFM.

Unlike the classifier-surrogate sweeps, this benchmark starts from a known exact
joint P*(A,B|X), derives its four mutually compatible views, and then perturbs
those views independently.  This isolates *which source of probability error*
makes hard marginal preservation, soft reconciliation, pooling, or direction
selection preferable.

The benchmark is checkpoint-free but directly targets the reconciliation
mechanism.  Adaptive choices are made from sampled validation labels; all test
scores are exact cross-entropies / distances to the known conditional joint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from core_tfm.data.synthetic import make_multiclass_dgp
from core_tfm.metrics.distributions import total_variation, marginal_distortion
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool, independent_joint
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.soft import soft_reconciliation
from core_tfm.reconciliation.weighted import (
    select_original_order,
    select_geometric_pool_weight,
    select_soft_hyperparameters,
)

EPS = 1e-12


@dataclass(frozen=True)
class Regime:
    name: str
    sigma_pa: float
    sigma_pb: float
    sigma_a_given_b: float
    sigma_b_given_a: float


REGIMES = [
    Regime("uniform_low_noise", 0.15, 0.15, 0.15, 0.15),
    Regime("marginals_reliable", 0.08, 0.08, 0.55, 0.55),
    Regime("conditionals_reliable", 0.55, 0.55, 0.08, 0.08),
    Regime("j1_reliable", 0.55, 0.08, 0.08, 0.55),
    Regime("j2_reliable", 0.08, 0.55, 0.55, 0.08),
    Regime("all_noisy", 0.55, 0.55, 0.55, 0.55),
]


def _normalize(x: np.ndarray, axis: int) -> np.ndarray:
    x = np.maximum(x, EPS)
    return x / x.sum(axis=axis, keepdims=True)


def _perturb_probs(p: np.ndarray, sigma: float, rng: np.random.Generator, axis: int) -> np.ndarray:
    if sigma == 0:
        return p.copy()
    logits = np.log(np.maximum(p, EPS)) + sigma * rng.normal(size=p.shape)
    logits -= logits.max(axis=axis, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=axis, keepdims=True)


def _views_from_joint(q: np.ndarray):
    pa = q.sum(axis=2)
    pb = q.sum(axis=1)
    agb = q / np.maximum(pb[:, None, :], EPS)
    bga = q / np.maximum(pa[:, :, None], EPS)
    agb = _normalize(agb, axis=1)
    bga = _normalize(bga, axis=2)
    return pa, pb, agb, bga


def _sample_labels(q: np.ndarray, rng: np.random.Generator):
    n, ka, kb = q.shape
    flat = q.reshape(n, -1)
    idx = np.array([rng.choice(ka * kb, p=flat[i]) for i in range(n)])
    return idx // kb, idx % kb


def _expected_nll(pstar: np.ndarray, q: np.ndarray) -> float:
    return float(-np.mean(np.sum(pstar * np.log(np.maximum(q, EPS)), axis=(1, 2))))


def _expected_brier(pstar: np.ndarray, q: np.ndarray) -> float:
    # E_Y~P* ||q-e_Y||^2 = ||q||^2 - 2<P*,q> + 1
    return float(np.mean(np.sum(q * q - 2.0 * pstar * q, axis=(1, 2)) + 1.0))


def _js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(np.maximum(p, EPS)) - np.log(np.maximum(m, EPS))), axis=(1, 2))
    kl_qm = np.sum(q * (np.log(np.maximum(q, EPS)) - np.log(np.maximum(m, EPS))), axis=(1, 2))
    return 0.5 * (kl_pm + kl_qm)


def _holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    out = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        out[idx] = running
    return out.tolist()


def run_seed(regime: Regime, seed: int, n: int, d: int, k: int, gamma: float):
    dgp = make_multiclass_dgp(n=n, d=d, k_a=k, k_b=k, gamma=gamma, nonlinear=True, seed=seed)
    truth = dgp.true_joint
    pa, pb, agb, bga = _views_from_joint(truth)
    rng = np.random.default_rng(90_000 + seed * 31 + sum(ord(c) for c in regime.name))

    pa_hat = _perturb_probs(pa, regime.sigma_pa, rng, axis=1)
    pb_hat = _perturb_probs(pb, regime.sigma_pb, rng, axis=1)
    agb_hat = _perturb_probs(agb, regime.sigma_a_given_b, rng, axis=1)
    bga_hat = _perturb_probs(bga, regime.sigma_b_given_a, rng, axis=2)

    j1 = pb_hat[:, None, :] * agb_hat
    j2 = pa_hat[:, :, None] * bga_hat

    ids = rng.permutation(n)
    n_val = max(200, int(0.25 * n))
    va, te = ids[:n_val], ids[n_val:]
    yva_a, yva_b = _sample_labels(truth[va], rng)

    order_sel = select_original_order(j1[va], j2[va], yva_a, yva_b)
    pool_sel = select_geometric_pool_weight(j1[va], j2[va], yva_a, yva_b)
    soft_sel = select_soft_hyperparameters(
        j1[va], j2[va], pa_hat[va], pb_hat[va], yva_a, yva_b,
        weights=(0.0, 0.25, 0.5, 0.75, 1.0),
        marginal_penalties=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
    )

    jt1, jt2 = j1[te], j2[te]
    pat, pbt, truth_t = pa_hat[te], pb_hat[te], truth[te]
    methods: dict[str, np.ndarray] = {
        "j1": jt1,
        "j2": jt2,
        "selected_order": jt1 if order_sel.name == "j1_b_then_a" else jt2,
        "independent": independent_joint(pat, pbt),
        "arithmetic": arithmetic_pool(jt1, jt2),
        "geometric": geometric_pool(jt1, jt2),
        "weighted_geometric": geometric_pool(jt1, jt2, weight=pool_sel.weight),
        "hard_core": marginal_preserving_reconciliation(jt1, jt2, pat, pbt).joint,
        "soft_core_0.1": soft_reconciliation(jt1, jt2, pat, pbt, lambda_a=0.1, lambda_b=0.1).joint,
        "soft_core_1": soft_reconciliation(jt1, jt2, pat, pbt, lambda_a=1.0, lambda_b=1.0).joint,
        "soft_core_10": soft_reconciliation(jt1, jt2, pat, pbt, lambda_a=10.0, lambda_b=10.0).joint,
        "adaptive_soft": soft_reconciliation(
            jt1, jt2, pat, pbt,
            reference_weight=soft_sel.weight,
            lambda_a=soft_sel.marginal_penalty,
            lambda_b=soft_sel.marginal_penalty,
        ).joint,
    }

    base = {
        "regime": regime.name,
        "seed": seed,
        "n": n,
        "d": d,
        "k": k,
        "gamma": gamma,
        "factorization_tv": float(total_variation(jt1, jt2).mean()),
        "selected_order_name": order_sel.name,
        "selected_pool_weight": pool_sel.weight,
        "selected_soft_weight": soft_sel.weight,
        "selected_soft_lambda": soft_sel.marginal_penalty,
    }
    rows = []
    for name, q in methods.items():
        rows.append({
            **base,
            "method": name,
            "expected_nll": _expected_nll(truth_t, q),
            "expected_brier": _expected_brier(truth_t, q),
            "tv_to_truth": float(total_variation(truth_t, q).mean()),
            "js_to_truth": float(_js(truth_t, q).mean()),
            "marginal_distortion": float(marginal_distortion(q, pat, pbt).mean()),
        })
    return rows


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        frame.groupby(["regime", "method"], as_index=False)
        .agg(
            expected_nll_mean=("expected_nll", "mean"),
            expected_nll_se=("expected_nll", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            tv_to_truth_mean=("tv_to_truth", "mean"),
            js_to_truth_mean=("js_to_truth", "mean"),
            marginal_distortion_mean=("marginal_distortion", "mean"),
        )
    )

    tests = []
    prespecified = [
        ("hard_core", "arithmetic"),
        ("adaptive_soft", "arithmetic"),
        ("adaptive_soft", "hard_core"),
        ("adaptive_soft", "selected_order"),
    ]
    for regime, sub in frame.groupby("regime"):
        wide = sub.pivot(index="seed", columns="method", values="expected_nll")
        local = []
        for a, b in prespecified:
            diff = wide[a] - wide[b]
            try:
                stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p = np.nan, 1.0
            local.append({
                "regime": regime,
                "method_a": a,
                "method_b": b,
                "mean_diff_a_minus_b": float(diff.mean()),
                "median_diff": float(diff.median()),
                "wins_a": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "n_seeds": int(len(diff)),
                "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                "p_raw": float(p),
            })
        adj = _holm_adjust([r["p_raw"] for r in local])
        for r, p_adj in zip(local, adj):
            r["p_holm"] = p_adj
            tests.append(r)
    return summary, pd.DataFrame(tests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--output", default="results/exact_view_perturbation.csv")
    args = ap.parse_args()

    rows = []
    for regime in REGIMES:
        print(f"regime={regime.name}", flush=True)
        for seed in range(args.seeds):
            rows.extend(run_seed(regime, seed, args.n, args.d, args.k, args.gamma))
    frame = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    summary, tests = summarize(frame)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    tests.to_csv(out.with_name(out.stem + "_tests.csv"), index=False)
    print("\nBest method per regime by exact expected NLL:")
    best = summary.loc[summary.groupby("regime")["expected_nll_mean"].idxmin(), ["regime", "method", "expected_nll_mean"]]
    print(best.to_string(index=False))
    print("\nPrespecified paired tests:")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
