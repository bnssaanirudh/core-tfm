# Reliability-Aware CoRe-TFM Revision Plan

This document converts the broader idea space into a controlled manuscript plan. It is deliberately conservative about unexecuted results.

## Revised scientific thesis

The target story is:

**inconsistency -> reliability diagnosis -> safe reconciliation decision**

rather than:

**inconsistency -> reconciliation always improves prediction**.

The revised manuscript should distinguish four properties throughout:

1. compatibility/coherence among probability views;
2. calibration;
3. proper-score predictive accuracy; and
4. downstream decision quality.

## Main-paper additions

### 1. Formal counterexamples: inconsistency is not an accuracy signal

Add a proposition/counterexample section proving that factorization-TV magnitude alone cannot determine whether reconciliation improves a proper score. The accompanying executable construction is implemented in `core_tfm.research_extensions.inconsistency_accuracy_counterexamples`.

Required paper claim after verification:

> Compatibility defect is a structural diagnostic, not a monotone surrogate for predictive regret or reconciliation benefit.

### 2. Explain the TabICLv2 / TabPFN-3 sign reversal

Fresh runs must archive direct marginal and conditional NLL/Brier/ECE, per-class calibration, entropy, factorization-TV and marginalization defects. Test whether reconciliation benefit is predicted by relative reliability of direct marginals versus conditional views.

Do not promote the current model interaction from exploratory to confirmatory until it survives independent sampling seeds.

### 3. Oracle opportunity vs selection regret

For each dataset/model/fold, report:

- available opportunity = arithmetic NLL - candidate-family oracle NLL;
- selection regret = selected NLL - candidate-family oracle NLL;
- selected-vs-arithmetic effect.

This separates lack of candidate headroom from failure of validation selection.

### 4. Safe Selective CoRe / complexity-aware selection

Evaluate nested policy families and a structural-risk penalty depending on validation size and family cardinality. All family/penalty choices must be made without test leakage. The goal is a principled fallback to arithmetic/simple pooling under severe validation scarcity.

### 5. Inference instability as a reliability signal

For each four-view query, generate repeated predictions under admissible inference perturbations such as feature permutations and context resampling. Archive Jensen-Shannon prediction dispersion. First test whether dispersion predicts held-out view loss. Only if validated should it be used for weighting/reconciliation strength.

Use the terminology `inference instability` or `predictive dispersion`, not `epistemic uncertainty`, unless an uncertainty interpretation is separately justified.

### 6. Rare-class sensitivity and support-adaptive penalties

Report rare-class exclusion sensitivity. Evaluate a simple support-adaptive marginal penalty as an exploratory method. Do not introduce unconstrained per-cell tuning in the main paper without a separate sample-complexity analysis.

### 7. Downstream decision utility on known-truth tasks

Use controlled tasks where P* is known. Sample or define utility matrices and compare expected decision regret for raw factorizations, pooling and reconciliation. This tests whether coherence changes actual decision quality rather than only NLL.

### 8. Policy transfer

Evaluate leave-one-dataset-out, model-global and universal-global policy transfer. This tests whether reconciliation behavior is entirely dataset-specific or whether model-level reliability patterns transfer.

## Required robustness before submission

- Five independent constrained-sampling seeds.
- Context-size sensitivity: 64/128/256/512/1024 where supported.
- Third released TFM after operational preflight.
- Rare-class sensitivity.
- Complete per-view reliability archive.
- Candidate-level validation score traces.
- Dataset-blocked inference retained as the primary real-data statistical unit.

## Framing changes

The abstract/introduction should make the negative bounded real result central rather than apologetic. The intended conclusion is conditional:

> Reconciliation can improve probabilistic prediction when the reliability structure and validation budget justify it, but structural coherence alone is insufficient and simple pooling can dominate complex repair under realistic low-validation regimes.

Candidate titles:

- `On Reconciling Inconsistent Probability Views from Tabular Foundation Models`
- `When Should Inconsistent Predictions Be Reconciled? Reliability-Aware Probability Reconciliation for Tabular Foundation Models`
- `Coherence Is Not Accuracy: Reliability-Aware Reconciliation of Tabular Foundation Model Predictions`

The final title should be chosen after the new robustness runs determine how strong the safe-selector and reliability-prediction results are.

## Positioning additions

Expand related work around:

- probabilistic opinion pooling and KL barycenters;
- stacking/model averaging;
- expert aggregation and exponential weighting;
- mixture-of-experts/gating;
- conditional compatibility and graphical models.

Explicitly state that CoRe candidate distributions are not independent experts; they are overlapping probability views/factorizations generated by the same frozen predictor.

## Failure/misuse guidance

Add a short `When not to reconcile` subsection:

- validation budget too small relative to policy-family complexity;
- severe rare-class scarcity without sensitivity analysis;
- no evidence that the targeted view is more reliable than the baseline pool;
- downstream utility highly sensitive to poorly calibrated tails/classes;
- inconsistency is small/large but there is no demonstrated predictive or utility headroom.

## Engineering/practical additions

- expose a lightweight consistency-audit API;
- report end-to-end runtime and memory, not only Sinkhorn microbenchmarks;
- publish negative/ablation tables as first-class supplementary artifacts;
- archive environment, source hashes, fold manifests and candidate score traces for all new runs.

## Follow-up paper directions (not required for current submission)

### Instance-adaptive learned reconciliation

Learn per-example direction weights and marginal penalties from entropy, agreement, support, calibration proxies and inference dispersion using cross-fitting/meta-learning.

### Multi-target graphical CoRe

Use pairwise four-view reconciliation on a Chow-Liu/tree structure and enforce shared node marginals. A tree-structured joint can avoid full exponential state enumeration while exposing local/global consistency questions.

### Online CoRe

Adapt policy weights as post-deployment labels arrive, connecting validation-size analysis to online expert aggregation/bandit methods.

### Conformalized CoRe

Study finite-sample predictive-set coverage after reconciliation; treat this as a separate guarantee from probability compatibility.

### Wasserstein CoRe

Investigate optimal-transport objectives only for ordinal or semantically embedded categories where a defensible ground cost exists.

## Implementation map

- Notebook: `notebooks/CoRe_TFM_RELIABILITY_AWARE_EXPERIMENTS.ipynb`
- Archive-derived runner: `experiments/run_reliability_aware_suite.py`
- Research utilities: `src/core_tfm/research_extensions.py`
- Experiment matrix: `configs/reliability_aware_experiments.yaml`
- Tests: `tests/test_research_extensions.py`

No new TFM benchmark result should enter `main.tex` until its evidence artifact is generated and validated.
