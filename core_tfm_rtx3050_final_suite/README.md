# CoRe-TFM — RTX 3050 Final Paper Experiment Suite

This ZIP is a **drop-in local runner** for `bnssaanirudh/core-tfm`. It is designed for the user's Windows 11 laptop with an **NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), Ryzen 7 6800HS and 16 GB RAM**.

It deliberately **reuses the repository's proven Q1 benchmark notebook and `core_tfm.robustness_runner.patch_q1_notebook`** instead of reimplementing the scientific benchmark. The only benchmark changes are the declared robustness parameters (seed/context size/output location/time budget) and replacement of the Colab setup cell with a local CUDA setup.

## What this suite covers

The single pipeline covers the final evidence groups we identified before paper freeze:

1. **Multi-seed robustness**
   - seeds: `11, 23, 42, 71, 101`
   - train/context limit: `256`
   - test limit: `128`
   - same 10 datasets, 5 outer folds, TabICLv2 + TabPFN-3, CatBoost boundary baseline.

2. **Context-size sensitivity**
   - train/context sizes: `64, 128, 256, 512, 1024`
   - seeds: `23, 42, 71`
   - the `256` runs are reused from the multi-seed matrix, so the suite performs **17 unique real-TFM variants rather than 20 duplicated variants**.

3. **Selector complexity / Safe-Selective evidence**
   - keeps the Q1 notebook's `selection_ablations.csv` enabled during the same fitted-view inference.
   - keeps `validation_fraction_sensitivity.csv` enabled.
   - separately runs the repository's controlled Safe Selective CoRe experiment.
   - no test labels are introduced into model/candidate selection by this wrapper.

4. **View-reliability analysis**
   - saves and aggregates the Q1 fold metrics.
   - derives direct marginal NLLs exactly from the raw factorizations:
     - `NLL(p_B) = NLL(J1) - NLL(p_A | B)`
     - `NLL(p_A) = NLL(J2) - NLL(p_B | A)`
   - combines these with conditional NLLs, factorization TV and Selective-vs-Arithmetic effects.
   - produces per-model and per-context reliability summaries/correlations.

5. **Rare-class sensitivity**
   - reloads the pinned pair datasets through `core_tfm.data.openml.load_pair_dataset`.
   - audits minimum target-class support for all 10 datasets.
   - recomputes the main 256-context Selective-vs-Arithmetic conclusion after minimum-support thresholds `1, 2, 5, 10`.

6. **Evidence integrity / paper freeze**
   - CUDA/hardware preflight.
   - repository test suite before expensive runs.
   - source commit and environment metadata per run.
   - resumable output folders and `COMPLETE.json` markers.
   - final completion gate.
   - SHA-256 manifest + final paper evidence ZIP.

## Important scope note

This ZIP covers the **five mandatory robustness gates** in the current CoRe-TFM protocol. A **third released TFM** is not silently added because the repository currently has no approved third-TFM checkpoint in the frozen protocol and TabFM was operationally excluded. Adding a third TFM requires a separate outcome-blind model preflight/protocol amendment rather than changing the final benchmark inside this runner.

## Expected folder layout

Extract this folder **inside the cloned repository**:

```text
core-tfm/
├─ .git/
├─ src/
├─ experiments/
├─ notebooks/
├─ results/
└─ core_tfm_rtx3050_final_suite/
   ├─ setup_windows.ps1
   ├─ run_final_suite.ps1
   ├─ ...
```

## 1. Clone/update the repository

If you do not already have it locally:

```powershell
git clone https://github.com/bnssaanirudh/core-tfm.git
cd core-tfm
```

Extract this ZIP so that `core_tfm_rtx3050_final_suite` is directly under `core-tfm`.

For publication runs, do **not** keep pulling/changing source halfway through the matrix. Every run freezes the current Git commit in its output metadata.

## 2. One-time environment setup

Open PowerShell:

```powershell
cd C:\path\to\core-tfm\core_tfm_rtx3050_final_suite
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

The setup creates `core-tfm\.venv`, installs the editable project, the repo's pinned TabICL/CatBoost versions and a CUDA-enabled PyTorch build.

The supplied Windows setup uses PyTorch `2.11.0` CUDA `12.8` wheels. If your NVIDIA driver is too old for that wheel, update the NVIDIA driver before changing the experimental environment.

## 3. Set TabPFN token

In the **same PowerShell window**:

```powershell
$env:TABPFN_TOKEN = "YOUR_TABPFN_TOKEN"
```

Do not save the token in this ZIP, the repository, notebooks, logs or Git commits.

Your Prior Labs account must also have the TabPFN-3 license accepted.

## 4. Run preflight

```powershell
.\preflight_windows.ps1
```

It verifies:

- CUDA is visible to PyTorch;
- GPU name/VRAM;
- a real CUDA matrix multiplication;
- `core_tfm`, TabICL, TabPFN, CatBoost, nbformat and nbclient imports;
- presence of `TABPFN_TOKEN`;
- Git source commit;
- installed package versions.

Output:

```text
results/rtx3050_final_suite_v1/PREFLIGHT.json
```

Do not start the paper matrix until this says `PASS`.

## 5. Run everything with one command

```powershell
.\run_final_suite.ps1
```

The pipeline executes in this order:

```text
preflight
  ↓
repo tests + controlled Safe Selective experiment
  ↓
5-seed real-TFM matrix
  ↓
context-size real-TFM matrix
  ↓
selector ablation + validation-sensitivity aggregation
  ↓
view-reliability analysis
  ↓
rare-class threshold sensitivity
  ↓
completion gate
  ↓
final evidence ZIP
```

## 6. Resume after interruption/reboot

The benchmark writes fold checkpoints. Completed variants are detected and skipped.

After opening a new PowerShell window, set the token again and run:

```powershell
cd C:\path\to\core-tfm\core_tfm_rtx3050_final_suite
$env:TABPFN_TOKEN = "YOUR_TABPFN_TOKEN"
.\resume_final_suite.ps1
```

Do not delete partially completed run directories when resuming.

## Useful partial commands

Multi-seed only:

```powershell
.\run_gpu_matrix.ps1 -Profile multi_seed
```

Context-size only:

```powershell
.\run_gpu_matrix.ps1 -Profile context
```

One variant:

```powershell
.\run_one_variant.ps1 -Seed 23 -TrainLimit 128 -Group context
```

Only regenerate analyses from existing runs:

```powershell
.\run_postprocessing.ps1
```

See `COMMANDS.txt` for direct Python equivalents.

## RTX 3050 4-GB behavior

The runner is deliberately conservative for low VRAM:

- only one experiment variant is executed at a time;
- model cache is shared across runs;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set;
- no parallel GPU workers are launched;
- CUDA cache is cleared between variants;
- finished folds are checkpointed;
- 512/1024 context cells that genuinely fail with CUDA OOM are recorded explicitly rather than being silently discarded.

Do not increase TabICL estimator count, batch sizes, test rows or launch multiple copies of the runner on a 4-GB GPU.

If a **256-context primary run** cannot complete because of VRAM, the final gate will remain incomplete; do not report it as a finished robustness study. The package intentionally does not silently change the model/protocol to make a desired result fit the laptop.

## Output structure

```text
results/rtx3050_final_suite_v1/
├─ PREFLIGHT.json
├─ MATRIX_STATUS.json
├─ multi_seed/
│  ├─ seed_11_train_256/
│  ├─ seed_23_train_256/
│  ├─ seed_42_train_256/
│  ├─ seed_71_train_256/
│  └─ seed_101_train_256/
├─ context/
│  ├─ seed_23_train_64/
│  ├─ seed_23_train_128/
│  ├─ seed_23_train_512/
│  ├─ ...
├─ cpu_experiments/
│  └─ safe_selective_controlled/
├─ analysis/
│  ├─ all_fold_results.csv
│  ├─ multi_seed_dataset_effects.csv
│  ├─ multi_seed_summary.json
│  ├─ context_summary.csv
│  ├─ model_effects.csv
│  ├─ all_selection_ablations.csv
│  ├─ all_validation_fraction_sensitivity.csv
│  ├─ view_reliability.csv
│  ├─ view_reliability_summary.csv
│  ├─ view_reliability_correlations.json
│  └─ rare_class/
│     ├─ dataset_support.csv
│     └─ threshold_effects.csv
├─ FINAL_COMPLETION_GATE.json
├─ PAPER_BUNDLE_MANIFEST.json
└─ CoRe_TFM_FINAL_PAPER_EVIDENCE.zip
```

Each real-TFM run directory also retains:

- `generated_variant.ipynb`
- `executed_variant.ipynb`
- `fold_results.csv`
- `selection_ablations.csv`
- `validation_fraction_sensitivity.csv`
- `fold_failures.json` when applicable
- `environment_metadata.json`
- `frozen_source_revisions.json`
- `RUN_STATUS.json`
- `COMPLETE.json` only when the expected benchmark result grid is complete.

## What to send back for paper writing

After the pipeline finishes, send back:

```text
results/rtx3050_final_suite_v1/CoRe_TFM_FINAL_PAPER_EVIDENCE.zip
```

If the completion gate fails, also send:

```text
FINAL_COMPLETION_GATE.json
MATRIX_STATUS.json
```

Those files identify exactly which experimental cells are complete, failed, or operationally unsupported.
