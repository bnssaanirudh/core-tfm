# CoRe-TFM

Research implementation for **post-hoc reconciliation of incompatible probability views from tabular foundation models (TFMs)**.

## Paper status

The current empirical release is the completed **S2 protocol** in
[`CoReTFM_Essential_Results_Notebooks_Codes_2026-10-21/`](CoReTFM_Essential_Results_Notebooks_Codes_2026-10-21/): 10 datasets × 2
TFMs × 4 held-out split settings × 2 training caps (128 and 256), for **160
configurations**. It includes 160 NPZ prediction caches, 160 JSON sidecars, and
40 frozen split specifications. The main setting uses a 52-example validation
selection budget and 128 test examples. The S2 data release is published at
[Zenodo DOI 10.5281/zenodo.23507196](https://doi.org/10.5281/zenodo.23507196).

The package also contains **six targeted clean replication runs**: three datasets
(Car, Nursery, Diamonds) × two TFMs, at seed 23 and training cap 256. These runs
are separate clean environment reruns, not six repeats of the full 160-configuration
matrix. The archive's current-status note explains how to read its preserved
historical reports and README files.

The S2 evidence collection, checksum manifest, and public data release are complete.
This describes the research evidence status; journal-specific author details and
submission checks remain tracked in [`paper/jmlr/SUBMISSION_CHECKLIST.md`](paper/jmlr/SUBMISSION_CHECKLIST.md).
The earlier five-fold benchmark under `results/q1_fast_complete_256_v1/` is retained
as a historical study, not the current S2 result set.

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

The earlier five-fold real benchmark remains archived for context. Current S2
configuration-level results and independent audit summaries are in the dated
evidence package linked above; S2 uses four split settings per dataset, which are
repeated tests within datasets rather than independent dataset-level units.

## Reliability-aware research extension

The next-stage research branch reframes the paper around **when reconciliation is
statistically justified**, rather than treating reconciliation as a universal
post-processing step. The runnable entry point is:

[`notebooks/CoRe_TFM_RELIABILITY_AWARE_EXPERIMENTS.ipynb`](notebooks/CoRe_TFM_RELIABILITY_AWARE_EXPERIMENTS.ipynb)

The reusable analysis runner is:

```bash
PYTHONPATH=src python experiments/run_reliability_aware_suite.py \
  --fold-results results/q1_fast_complete_256_v1/fold_results.csv \
  --output results/reliability_aware_v1
```

Implemented/scaffolded additions include:

- analytic counterexamples showing that factorization TV is not an accuracy signal;
- inconsistency-versus-reconciliation-gain correlation analysis;
- oracle opportunity versus validation-selection-regret decomposition;
- model-specific raw-view reliability diagnostics for the TabICLv2/TabPFN-3 reversal;
- structural-risk/complexity penalties for **Safe Selective CoRe**;
- Jensen-Shannon inference-dispersion utilities for future label-free reliability weighting;
- support-adaptive rare-class penalty utilities;
- policy-transfer and candidate-family headroom analyses;
- known-truth downstream decision-regret evaluation utilities;
- a canonical experiment matrix for independent sampling seeds, context sizes,
  a third TFM, per-view calibration archives, and inference-perturbation studies.

The full experiment specification is in
[`configs/reliability_aware_experiments.yaml`](configs/reliability_aware_experiments.yaml).
Fresh-inference experiments are intentionally marked as pending; no unexecuted
result is represented as established evidence.

Multi-target graphical reconciliation, online/streaming adaptation, conformalized
reconciliation, and Wasserstein objectives are retained as follow-up research
tracks rather than being mixed into the main submission before the pairwise
reliability story is complete.

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
PYTHONPATH=src python experiments/run_reliability_aware_suite.py
```

Validate every S2 cache and JSON sidecar against the package's uploaded SHA-256
manifests (including the six clean reruns) with:

```bash
python experiments/validate_s2_evidence.py
```

## Research guardrail

Any normalized joint is internally coherent. Therefore, **zero inconsistency after reconciliation is not treated as evidence of usefulness**. The paper evaluates proper scoring, conditional prediction, calibration, distortion, dependence fidelity, truth distance when available, decision utility when truth is available, and validation-selected consistency tax/dividend.
