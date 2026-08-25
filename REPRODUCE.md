# Reproduction guide

## Core method validation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[test]'
pytest
PYTHONPATH=src python experiments/smoke_synthetic.py
```

The smoke experiment uses an inner validation split for adaptive weighting and a disjoint test split for all reported scores.

## Completed bounded-context benchmark

The exact Colab notebook is
`notebooks/CoRe_TFM_Q1_FAST_COMPLETE_256_Colab.ipynb`. Its evidence archive is
expanded under `results/q1_fast_complete_256_v1/`. The completed profile uses:

- ten datasets and five deterministic outer folds;
- at most 256 training, 52 validation, and 128 test observations per fold;
- TabICLv2 2.1.1 with two estimators and disabled KV cache;
- TabPFN 8.4.0 as the TabPFN-3 interface; and
- CatBoost 1.2.10 as a boundary baseline, excluded from TFM-only inference.

TabPFN weights require prior acceptance of the model license and a valid
`TABPFN_TOKEN`. Never store that token in the notebook, logs, or repository.

Validate the committed matrix and reproduce the dataset-blocked statistics with:

```bash
PYTHONPATH=src python experiments/analyze_q1_benchmark.py
```

The expected primary joint-NLL effect is Selective CoRe minus arithmetic pooling
`+0.0018312632`, based on ten dataset units after averaging TabICLv2 and TabPFN-3.
Negative values would favor Selective CoRe; this completed result therefore favors
arithmetic pooling.

## Reliability-aware extension: immediately runnable

The extension notebook is:

`notebooks/CoRe_TFM_RELIABILITY_AWARE_EXPERIMENTS.ipynb`

It reuses the archived benchmark and does not require downloading TFM checkpoints for
its first analysis block. Run the equivalent CLI pipeline with:

```bash
PYTHONPATH=src python experiments/run_reliability_aware_suite.py \
  --fold-results results/q1_fast_complete_256_v1/fold_results.csv \
  --output results/reliability_aware_v1
```

This writes only derived results into `results/reliability_aware_v1/` and never
modifies the frozen Q1 evidence package. Outputs include:

- oracle opportunity vs validation-selection-regret decomposition;
- factorization-TV vs reconciliation-gain correlations;
- analytic counterexamples showing that inconsistency is not an accuracy metric;
- model-specific raw-view reliability proxies;
- candidate-family headroom summaries;
- method-transfer proxies across datasets/models.

The notebook also demonstrates reusable utilities for inference-dispersion weighting,
support-adaptive rare-class penalties, structural-risk complexity penalties, and
known-truth decision regret. Those demonstrations are mechanisms, not new empirical
claims about the TFMs.

## Fresh-inference experiment protocol

Canonical settings are in `configs/reliability_aware_experiments.yaml`. Before a
revised journal submission, run and archive the following as a new evidence package:

1. five independent constrained-sampling seeds across the ten datasets and five folds;
2. context-size sensitivity at 64, 128, 256, 512, and 1,024 examples where supported;
3. at least one third released TFM after an operational preflight;
4. direct-marginal and conditional NLL/Brier/ECE plus per-class reliability metrics;
5. per-example factorization TV, marginalization defects, predictive entropy, and
   inference dispersion under admissible perturbations;
6. complete validation candidate-score traces, enabling Safe Selective CoRe to be
   evaluated without test leakage;
7. rare-class exclusion and support-adaptive penalty sensitivity;
8. leave-one-dataset-out/model-global policy-transfer evaluation; and
9. downstream expected-utility regret on controlled tasks where the true joint is known.

Do not reuse the current test set to tune candidate-family complexity, dispersion
temperatures, class-adaptive penalties, or any new policy. These must be chosen in an
inner validation/cross-fitting layer or pre-registered before comparative test scores
are examined.

## Follow-up research kept outside the main submission

The current extension intentionally does not mix in every promising direction. The
following are tracked as separate follow-up projects unless the pairwise reliability
story is complete first:

- Chow-Liu/tree-structured multi-target reconciliation;
- learned per-instance gating/meta-reconciliation;
- online/streaming policy adaptation;
- conformalized reconciled predictive sets; and
- Wasserstein/optimal-transport reconciliation for ordinal or embedded categories.

## Reproduction caveat

The source TFM-consistency study specifies five-fold cross-testing but does not
publish a split seed. CoRe-TFM therefore uses deterministic stratified splits and
records them explicitly. The completed run is evidence about a bounded small-context
deployment regime, not a replacement for the earlier full-training Car and Wine
pilots or an exact numerical replication of the source study.
