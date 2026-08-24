# CoRe-TFM

Research implementation for **post-hoc reconciliation of incompatible probability views from tabular foundation models (TFMs)**.

## Paper status

The repository now contains a complete bounded-context benchmark: 10 datasets,
five outer folds, two primary released TFMs (TabICLv2 and TabPFN-3), and CatBoost
as a non-TFM boundary baseline. Every fold uses at most 256 training, 52 validation,
and 128 test observations. The evidence package, exact notebook, protocol amendment,
checksums, environment record, and generated plots are under
[`results/q1_fast_complete_256_v1/`](results/q1_fast_complete_256_v1/) and
[`notebooks/`](notebooks/).

The result is scientifically informative but not a performance win. Averaging the
two TFMs within each dataset, Selective CoRe is worse than arithmetic pooling by
`+0.001831` joint NLL (95% bootstrap CI `[+0.000548, +0.003301]`; 2/10 wins;
two-sided Wilcoxon `p=0.02734`). It reduces marginal distortion by `0.003855` but
worsens Brier score by `0.000646`. The manuscripts therefore frame CoRe-TFM as a
diagnostic reliability and consistency--accuracy trade-off, not as an unconditional
improvement over pooling.

The draft remains **not submission-ready** until sampling-seed and training-size
sensitivity are completed, the rare-class dataset choice is resolved, and author
metadata/disclosures are finalized. A third released TFM is strongly recommended
for broad cross-TFM claims; CatBoost is not treated as its substitute.

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
3. **Heterogeneous task mixtures** in which view reliability changes by task. In
   the completed 100-task study, Selective CoRe beats the best fixed method,
   arithmetic pooling, by `0.006660` mean exact NLL, wins 68/100 paired tasks,
   and has Wilcoxon `p=1.25e-11`. Its mean regret to the non-deployable full-family
   oracle is `0.000710`. Controlled validation-size experiments reduce oracle regret
   from `0.005602` at 100 labels to `0.001155` at 800 labels.

The real benchmark confirms measurable incompatibility (mean factorization TV
`0.06249` across the primary TFMs) but shows that a 48-candidate selector can
overfit a 52-example validation set. Pool-only selection has slightly lower mean
NLL than the full family, while Hard CoRe removes marginal distortion at a clear
proper-score cost. Effects reverse by model: Selective CoRe is worse on TabICLv2
by `0.006694` and better on TabPFN-3 by `0.003031` on average.

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
PYTHONPATH=src python experiments/analyze_q1_benchmark.py
```

## Research guardrail

Any normalized joint is internally coherent. Therefore, **zero inconsistency after reconciliation is not treated as evidence of usefulness**. The paper evaluates proper scoring, conditional prediction, calibration, distortion, dependence fidelity, truth distance when available, and validation-selected consistency tax/dividend.
