# CoRe-TFM

Research implementation for **post-hoc reconciliation of incompatible probability views from tabular foundation models (TFMs)**.

## Paper status

The current master manuscript is a **JMLR-formatted working draft** in [`paper/jmlr/`](paper/jmlr/). GitHub Actions downloads the official JMLR `jmlr2e` style directly from the JMLR style repository and compiles the public draft PDF. A Springer *Machine Learning* adaptation is maintained as a secondary target.

The draft is **not submission-ready yet** because the controlled experiments must still be complemented by the complete released-model benchmark on TabPFN-3, TabICLv2, and TabFM.

## Core problem

For two categorical targets, a frozen predictor can expose four views:

- `p(A | X)`
- `p(B | X)`
- `p(A | B, X)`
- `p(B | A, X)`

which induce

`J1(A,B|X) = p(B|X) p(A|B,X)` and
`J2(A,B|X) = p(A|X) p(B|A,X)`.

CoRe-TFM asks which coherent joint, if any, should replace incompatible views **without confusing coherence with calibration or predictive correctness**.

## Implemented methods

- arithmetic and geometric KL-barycenter baselines
- Hard CoRe / marginal-preserving KL projection
- Soft CoRe with generalized Sinkhorn-style scaling
- validation-selected direction weighting and marginal penalty
- Selective CoRe: validation-gated choice among raw orders, pooling, and reconciliation
- joint/marginal/conditional proper scores and calibration diagnostics
- truth-distance and dependence-fidelity metrics for controlled experiments
- TabPFN-3, TabICLv2, and TabFM adapters

## Current empirical evidence

The repository now contains three complementary controlled studies:

1. **Surrogate DGP sweeps** with exact conditional truth across dependence, sample size, dimensionality, class cardinality, and imbalance.
2. **Exact-view perturbations** that independently corrupt the two marginals and two conditional families, showing when hard marginal preservation helps or hurts.
3. **Heterogeneous task mixtures** in which view reliability changes by task. In the current 60-task full-oracle study, Selective CoRe beats the best fixed method (arithmetic pooling) by about `0.00581` mean exact NLL, wins 41/60 paired tasks, and has Wilcoxon `p = 5.79e-7`. A separate validation-size study shows selection regret decreasing as held-out validation data increase.

The repository also contains a **five-fold released-model pilot** on TabICLv2 + UCI Car Evaluation. Mean factorization TV is `0.06931 ± 0.00354`; the best mean joint NLL is the unreconciled `B→A` chain (`1.12089`), ahead of geometric pooling (`1.13434`), Soft CoRe with `lambda=1` (`1.13840`), and Hard CoRe (`1.20412`). A leakage-free five-fold Selective CoRe run chooses the raw `B→A` direction in 4/5 folds and a mild Soft CoRe policy once, reaching `1.12116 ± 0.00485` mean joint NLL—only `0.00027` above the per-fold best original direction. This supports selective rather than unconditional repair. It is one model–dataset cell, not an average released-TFM performance claim.

## Reproduce core tests

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[test]'
pytest
```

Checkpoint-free smoke test:

```bash
PYTHONPATH=src python experiments/smoke_synthetic.py
```

Key empirical scripts:

```bash
PYTHONPATH=src python experiments/run_exact_view_perturbation.py
PYTHONPATH=src python experiments/run_selective_mixture.py
PYTHONPATH=src python experiments/run_selective_validation_size.py
```

## Research guardrail

Any normalized joint is internally coherent. Therefore, **zero inconsistency after reconciliation is not treated as evidence of usefulness**. The paper evaluates proper scoring, conditional prediction, calibration, distortion, dependence fidelity, truth distance when available, and validation-selected consistency tax/dividend.
