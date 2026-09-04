# Expected Outputs

The exact filenames can evolve with the repository runner, but a successful run
should contain the following categories under the selected output root.

## Per-variant evidence
- `fold_results.csv`
- `RUN_STATUS.json`
- `COMPLETE.json`
- `environment_metadata.json`
- `frozen_source_revisions.json`
- `selection_ablations.csv`
- `validation_fraction_sensitivity.csv`
- generated/executed notebook

## Aggregate evidence
Under `analysis/`:
- all fold results
- Selective-vs-arithmetic dataset/model summaries
- multi-seed summaries
- context-size summaries
- model effects
- view-reliability tables/correlations
- selection-ablation aggregation
- validation-sensitivity aggregation

## Rare-class evidence
Under `analysis/rare_class/`:
- dataset support table
- threshold/exclusion effects
- dataset/seed effects with support
- summary metadata

## CPU controlled experiment
Under `cpu_experiments/`:
- Safe-Selective controlled outputs
- frozen archive-derived reference analyses
- CPU completion metadata

## Final audit
Under `final_submission_audit/`:
- `frozen_evidence_audit.json`
- `FINAL_STATUS.json`
- `sha256_manifest.json`

A file being present is not by itself proof that the completion gate passed.
