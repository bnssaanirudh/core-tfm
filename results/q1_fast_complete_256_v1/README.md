# Bounded benchmark evidence package

This directory is the expanded contents of
`core_tfm_q1_fast_complete_256_v1_final.zip`. Files covered by
`sha256sums_final.json` are preserved byte-for-byte.

## Scope

- Protocol: `v7_bounded_256x128_complete_2026-08-23`
- 10 datasets x 3 active models x 5 folds
- Primary TFMs: TabICLv2 and TabPFN-3
- Boundary baseline: CatBoost (not a TFM)
- 256 training, 52 validation, and 128 test observations per fold
- 8 reported methods, 4 selection-family ablations, and 3 validation fractions

## Statistical correction

`table_statistics.tex` and `cell_level_statistics.csv` are retained as original
run artifacts, but their 30-cell Selective-versus-arithmetic row must not be called
the primary confirmatory analysis. Those cells share datasets, and one model is
CatBoost. The manuscript's primary analysis uses ten dataset units after averaging
TabICLv2 and TabPFN-3, as recorded in
`dataset_blocked_primary_inference.json` and reproduced by:

```bash
PYTHONPATH=src python experiments/analyze_q1_benchmark.py
```

The primary Selective-minus-arithmetic joint-NLL effect is `+0.0018312632`;
negative would favor Selective CoRe. The completed bounded result therefore favors
arithmetic pooling.

## Integrity

All 46 recorded hashes were independently checked before integration. The only
files not covered by the original checksum manifest are this explanatory README
and any later repository-level documentation.
