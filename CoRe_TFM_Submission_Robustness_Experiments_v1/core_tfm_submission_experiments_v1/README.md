# CoRe-TFM Submission Robustness Experiment Bundle v1

This bundle is an **experiment orchestrator**, not a result package.

It is designed to be extracted into the root of the current
`bnssaanirudh/core-tfm` repository. It deliberately reuses the repository's
existing `core_tfm_rtx3050_final_suite` runner instead of duplicating the model
inference implementation.

## Scientific rule

**Do not copy any number produced by a partial run into the paper.**
A new result is admissible only after:
1. the requested run matrix is complete or an operationally unsupported cell is
   explicitly logged;
2. post-processing completes;
3. the completion gate passes;
4. the final evidence manifest and checksums are generated.

The frozen Q1 benchmark remains read-only.

## Why these experiments?

The existing primary benchmark is already complete. The remaining paper-facing
questions are robustness questions:

1. **Independent sampling-seed stability**
   - Does the sign/magnitude of Selective CoRe minus arithmetic pooling survive
     independent constrained-sampling realizations?

2. **Training-context sensitivity**
   - Does the conclusion change as the bounded context grows?

3. **Rare-class sensitivity**
   - Are low-support datasets driving the conclusion?

4. **Safe / complexity-aware selection**
   - When the validation set is small, does a restricted candidate family or
     arithmetic fallback reduce selection regret?

5. **View-reliability diagnostics**
   - Are direct-marginal / conditional losses and factorization-TV associated
     with when reconciliation helps?

The bundle does **not** add a third TFM. If you submit without one, all empirical
cross-model claims must remain explicitly limited to TabICLv2 and TabPFN-3.

## Presets

### `minimal_submission.yaml`
Smallest serious paper-strength robustness package:
- independent seeds: 11, 23, 42
- baseline train limit: 256
- context sensitivity: 256, 512, 1024
- context seeds: 23, 42
- 5 folds
- test limit: 128
- selection ablations + validation sensitivity retained
- rare-class thresholds: 1, 2, 5, 10
- controlled Safe-Selective experiment: 100 tasks

The run-matrix code de-duplicates `(seed, train_limit)` combinations. This preset
therefore schedules **7 GPU notebook variants**:
- 3 multi-seed variants at train=256
- 4 additional context variants (23/42 x 512/1024)

### `full_protocol.yaml`
Matches the broader frozen robustness protocol:
- seeds: 11, 23, 42, 71, 101
- context sizes: 64, 128, 256, 512, 1024
- context seeds: 23, 42, 71
- same 5-fold / 128-test bounded protocol
- full post-processing and completion gate

This schedules **17 de-duplicated GPU notebook variants**.

## Expected repository structure

After extracting this ZIP:

```text
core-tfm/
├─ core_tfm_submission_experiments_v1/
├─ core_tfm_rtx3050_final_suite/
├─ experiments/
├─ notebooks/
├─ results/
└─ src/
```

## Windows / RTX workflow

From PowerShell in the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
$env:TABPFN_TOKEN="YOUR_EXISTING_VALID_TOKEN"

python core_tfm_submission_experiments_v1\scripts\preflight_submission.py --repo .
python core_tfm_submission_experiments_v1\scripts\plan_matrix.py --preset minimal

powershell -ExecutionPolicy Bypass -File `
  core_tfm_submission_experiments_v1\run_minimal_windows.ps1
```

To run the full protocol:

```powershell
powershell -ExecutionPolicy Bypass -File `
  core_tfm_submission_experiments_v1\run_full_windows.ps1
```

## Linux / CUDA workflow

```bash
export TABPFN_TOKEN="YOUR_EXISTING_VALID_TOKEN"
python core_tfm_submission_experiments_v1/scripts/preflight_submission.py --repo .
bash core_tfm_submission_experiments_v1/run_minimal_linux.sh
```

## Resuming

The underlying runner is resume-aware. Re-run the same command. Completed
variant directories are detected and not intentionally overwritten.

## Staged execution

You can run individual stages:

```powershell
python core_tfm_submission_experiments_v1\scripts\run_suite.py `
  --repo . --preset minimal --stage cpu

python core_tfm_submission_experiments_v1\scripts\run_suite.py `
  --repo . --preset minimal --stage gpu

python core_tfm_submission_experiments_v1\scripts\run_suite.py `
  --repo . --preset minimal --stage postprocess

python core_tfm_submission_experiments_v1\scripts\finalize_submission.py `
  --repo . --preset minimal
```

## Output roots

Minimal:
`results/core_tfm_submission_minimal_v1/`

Full:
`results/core_tfm_submission_full_v1/`

The frozen historical results directory
`results/q1_fast_complete_256_v1/`
must never be used as a new output root.

## What to put in the paper after completion

Use the generated analysis tables to answer only these questions:
- Is the dataset-blocked effect stable across independent seeds?
- Does it change with context size?
- Does removing low-support datasets change the conclusion?
- Does restricted/safe candidate selection reduce regret?
- Which measured view-reliability quantities correlate with repair benefit?

Do not convert exploratory cell-level patterns into confirmatory independent
samples. The primary real-data inferential unit remains the dataset.
