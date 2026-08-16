# CoRe-TFM

Research implementation for **post-hoc reconciliation of incompatible conditional predictions in tabular foundation models (TFMs)**.

## Current scope

Classification-only, pairwise target reconciliation. Given a frozen probabilistic classifier, the pipeline estimates:

- `p(A | X)`
- `p(B | X)`
- `p(A | B, X)` for every observed class of `B`
- `p(B | A, X)` for every observed class of `A`

and constructs the two implied joints

`J1(A,B|X) = p(B|X) p(A|B,X)` and
`J2(A,B|X) = p(A|X) p(B|A,X)`.

The core proposed method, **Marginal-Preserving Reconciliation (MPR)**, forms a geometric consensus and KL-projects it onto the transportation polytope with the TFM's direct marginals. This preserves direct marginal predictions exactly while repairing the dependence structure.

## Implemented

- Exact classification joint construction
- Factorization TV metric
- Arithmetic, geometric/log, and independence baselines
- Marginal-Preserving Reconciliation via IPF/Sinkhorn-style scaling
- Batched soft reconciliation with tunable marginal-fidelity penalties
- Validation-adaptive direction weighting and soft hyperparameter selection
- Reconciliation Distortion and Marginal Distortion
- Joint NLL, multiclass Brier score, and top-label ECE
- Binary and multiclass synthetic DGPs with known conditional joints
- Known-truth sweep runner
- Generic batched conditional-extraction pipeline
- Optional adapters for TabPFN-3, TabICLv2, and Google TabFM
- Pinned OpenML dataset specifications
- Unit tests and checkpoint-free synthetic experiments

## Install core

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[test]'
pytest
```

## Smoke experiment

```bash
python experiments/smoke_synthetic.py
```

This intentionally uses logistic regression so the reconciliation code can be validated without downloading a foundation-model checkpoint.

## Reproducibility target

The first external checkpoint sanity test is **Credit + TabPFN-3**, followed by five-fold evaluation and the full real-data benchmark.

## Research guardrail

A repaired joint is internally coherent by construction. The research contribution is therefore evaluated through **distortion, proper scoring rules, calibration, conditional performance, and distance to known synthetic ground truth**, rather than by claiming zero post-repair factorization inconsistency as the empirical result.

## Status

The method and synthetic evaluation pipeline are implemented. Real TabPFN-3, TabICLv2, and TabFM checkpoint experiments are the next validation stage.
