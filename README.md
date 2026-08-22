# CoRe-TFM

Research implementation for **post-hoc reconciliation of incompatible probability views from tabular foundation models (TFMs)**.

## Paper status

The evidence currently fits **Transactions on Machine Learning Research (TMLR)** best: its technical-correctness emphasis matches a rigorous method paper with two real TabICLv2 cells but without a full multi-model matrix. The repository retains a JMLR-formatted working draft in [`paper/jmlr/`](paper/jmlr/) and a Springer *Machine Learning* adaptation in [`paper/springer/`](paper/springer/) as synchronized alternatives. GitHub Actions compiles the JMLR draft with the official `jmlr2e` style.

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

The repository also contains **two five-fold released-model pilots** on TabICLv2 2.1.1. On UCI Car Evaluation, factorization TV is `0.06956 ± 0.00475`; the raw `B→A` chain has the best mean NLL (`1.11686 ± 0.00786`) and leakage-free Selective CoRe chooses it in 4/5 folds (`1.11691 ± 0.00786`). On Wine, reconstructed from canonical UCI red/white source files with recorded SHA-256 hashes, factorization TV is `0.02450 ± 0.00200`; arithmetic pooling has the best fixed mean NLL (`0.71678 ± 0.01085`), while Selective CoRe chooses arithmetic pooling in 2/5 folds and Soft CoRe in 3/5 (`0.71797 ± 0.01237`). Neither cell supports unconditional projection. These are two data-set cells for one released model, not an average released-TFM performance claim.

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
