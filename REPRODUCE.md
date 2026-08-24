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

## Required robustness extensions

Before journal submission, freeze a new protocol amendment and run at least three
independent constrained-sampling seeds. Add a training-context sensitivity on a
representative subset at 256, 512, 1,024, and full feasible training size. Resolve
the Marketing/rare-class design explicitly and report a sensitivity excluding
rare-class datasets. These extensions must be labeled confirmatory or exploratory
before inspecting their comparative scores.

## Reproduction caveat

The source TFM-consistency study specifies five-fold cross-testing but does not
publish a split seed. CoRe-TFM therefore uses deterministic stratified splits and
records them explicitly. The completed run is evidence about a bounded small-context
deployment regime, not a replacement for the earlier full-training Car and Wine
pilots or an exact numerical replication of the source study.
