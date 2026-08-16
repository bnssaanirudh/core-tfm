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

## Real TFM validation plan

The next empirical stage is five-fold evaluation with TabPFN-3, TabICLv2, and TabFM on the pinned OpenML classification pairs in `configs/real_datasets.yaml`.

The first external checkpoint sanity test is **Credit + TabPFN-3**. Exact package/checkpoint versions and model-weight licenses should be recorded with every reported result.

## Reproduction caveat

The source TFM-consistency study specifies five-fold cross-testing but does not publish a split seed in the manuscript. CoRe-TFM therefore uses deterministic stratified splits and records them explicitly. Exact numerical equality with the source paper is not expected unless its original fold assignment is released; the first goal is replication of the same non-zero consistency phenomenon and comparable magnitude.
