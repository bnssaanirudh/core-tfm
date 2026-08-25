"""Controlled Safe Selective CoRe experiment with validation-only family selection.

This experiment reuses the heterogeneous known-truth task generator. It records
all validation candidate scores, applies structural-risk penalties using only the
validation sample size and candidate-family size, freezes the chosen candidate,
and then evaluates exact expected test NLL under P*.

It is a controlled mechanism experiment. It must not be described as a fresh
real-TFM result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from core_tfm.data.synthetic import make_multiclass_dgp
from core_tfm.reconciliation.baselines import arithmetic_pool, geometric_pool
from core_tfm.reconciliation.mpr import marginal_preserving_reconciliation
from core_tfm.reconciliation.soft import soft_reconciliation
from core_tfm.reconciliation.selective import select_reconciliation_policy
from core_tfm.research_extensions import safe_select_from_scores

EPS = 1e-12
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def normalize(x, axis):
    x = np.maximum(x, EPS)
    return x / x.sum(axis=axis, keepdims=True)


def views(q):
    pa = q.sum(2)
    pb = q.sum(1)
    return pa, pb, normalize(q / np.maximum(pb[:, None, :], EPS), 1), normalize(q / np.maximum(pa[:, :, None], EPS), 2)


def perturb(p, sigma, rng, axis):
    z = np.log(np.maximum(p, EPS)) + sigma * rng.normal(size=p.shape)
    z -= z.max(axis=axis, keepdims=True)
    z = np.exp(z)
    return z / z.sum(axis=axis, keepdims=True)


def sample_y(q, rng):
    n, ka, kb = q.shape
    flat = q.reshape(n, -1)
    z = np.array([rng.choice(ka * kb, p=flat[i]) for i in range(n)])
    return z // kb, z % kb


def enll(p, q):
    return float(-np.mean(np.sum(p * np.log(np.maximum(q, EPS)), axis=(1, 2))))


def candidates(j1, j2, pa, pb):
    out = {
        "j1_b_then_a": j1,
        "j2_a_then_b": j2,
        "arithmetic": arithmetic_pool(j1, j2),
        "geometric[w=0.5]": geometric_pool(j1, j2),
        "mpr[w=0.5]": marginal_preserving_reconciliation(j1, j2, pa, pb).joint,
    }
    for w in WEIGHTS:
        if abs(w - 0.5) > 1e-15:
            out[f"geometric[w={w:g}]"] = geometric_pool(j1, j2, weight=w)
            out[f"mpr[w={w:g}]"] = marginal_preserving_reconciliation(j1, j2, pa, pb, reference_weight=w).joint
        for lam in LAMBDAS:
            out[f"soft[w={w:g},lambda={lam:g}]"] = soft_reconciliation(
                j1, j2, pa, pb, reference_weight=w, lambda_a=lam, lambda_b=lam
            ).joint
    return out


def family_scores(score_map):
    keys = list(score_map)
    raw_pool = {
        k: score_map[k]
        for k in keys
        if k in {"j1_b_then_a", "j2_a_then_b", "arithmetic"} or k.startswith("geometric[")
    }
    repair_small = {
        k: score_map[k]
        for k in keys
        if k == "arithmetic" or k.startswith("geometric[") or k in {"soft[w=0.5,lambda=0.3]", "soft[w=0.5,lambda=1]", "soft[w=0.5,lambda=3]"}
    }
    return {
        "fallback": {"arithmetic": score_map["arithmetic"]},
        "raw_pool": raw_pool,
        "repair_small": repair_small,
        "full": dict(score_map),
    }


def run_task(seed, n, d, k, gamma, beta, delta):
    dgp = make_multiclass_dgp(n=n, d=d, k_a=k, k_b=k, gamma=gamma, nonlinear=True, seed=seed)
    truth = dgp.true_joint
    pa, pb, agb, bga = views(truth)
    rng = np.random.default_rng(910000 + seed)
    sig = np.exp(rng.uniform(np.log(0.04), np.log(0.75), size=4))
    pah = perturb(pa, sig[0], rng, 1)
    pbh = perturb(pb, sig[1], rng, 1)
    agbh = perturb(agb, sig[2], rng, 1)
    bgah = perturb(bga, sig[3], rng, 2)
    j1 = pbh[:, None, :] * agbh
    j2 = pah[:, :, None] * bgah

    ids = rng.permutation(n)
    nv = max(250, int(0.30 * n))
    va, te = ids[:nv], ids[nv:]
    ya, yb = sample_y(truth[va], rng)

    vanilla = select_reconciliation_policy(
        j1[va], j2[va], pah[va], pbh[va], ya, yb,
        weights=WEIGHTS, marginal_penalties=LAMBDAS,
    )
    scores = vanilla.scores
    safe_candidate, safe_family, diagnostics = safe_select_from_scores(
        family_scores(scores), n_validation=len(va), fallback="arithmetic", delta=delta, beta=beta
    )

    test_candidates = candidates(j1[te], j2[te], pah[te], pbh[te])
    if safe_candidate not in test_candidates:
        raise KeyError(f"Safe candidate not found in test family: {safe_candidate}")

    pt = truth[te]
    safe_nll = enll(pt, test_candidates[safe_candidate])
    arithmetic_nll = enll(pt, test_candidates["arithmetic"])
    oracle_name = min(test_candidates, key=lambda m: enll(pt, test_candidates[m]))
    oracle_nll = enll(pt, test_candidates[oracle_name])

    row = {
        "task": seed,
        "n_validation": len(va),
        "beta": beta,
        "delta": delta,
        "safe_candidate": safe_candidate,
        "safe_family": safe_family,
        "safe_nll": safe_nll,
        "arithmetic_nll": arithmetic_nll,
        "oracle_method": oracle_name,
        "oracle_nll": oracle_nll,
        "safe_minus_arithmetic": safe_nll - arithmetic_nll,
        "safe_oracle_regret": safe_nll - oracle_nll,
        "sigma_pa": sig[0],
        "sigma_pb": sig[1],
        "sigma_a_given_b": sig[2],
        "sigma_b_given_a": sig[3],
    }
    for k2, v2 in diagnostics.items():
        row[f"diag_{k2}"] = v2
    return row, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--n", type=int, default=2200)
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--output-dir", type=Path, default=Path("results/safe_selective_controlled_v1"))
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    trace_rows = []
    for seed in range(a.tasks):
        if seed % 10 == 0:
            print("task", seed, flush=True)
        row, scores = run_task(seed, a.n, a.d, a.k, a.gamma, a.beta, a.delta)
        rows.append(row)
        for candidate, score in scores.items():
            trace_rows.append({"task": seed, "candidate": candidate, "validation_nll": score})

    df = pd.DataFrame(rows)
    traces = pd.DataFrame(trace_rows)
    df.to_csv(a.output_dir / "safe_selective_tasks.csv", index=False)
    traces.to_csv(a.output_dir / "validation_candidate_scores.csv", index=False)

    diff = df["safe_minus_arithmetic"].to_numpy()
    stat, p = wilcoxon(diff)
    summary = {
        "scope": "controlled_known_truth",
        "n_tasks": int(len(df)),
        "mean_safe_minus_arithmetic": float(diff.mean()),
        "safe_wins": int((diff < 0).sum()),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
        "mean_safe_oracle_regret": float(df["safe_oracle_regret"].mean()),
        "selection_counts": df["safe_candidate"].value_counts().to_dict(),
        "family_counts": df["safe_family"].value_counts().to_dict(),
        "guardrail": "All family/candidate choices use validation labels only; exact test truth is used only after selection is frozen.",
    }
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (a.output_dir / "COMPLETE.json").write_text(json.dumps({"complete": True, **summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
