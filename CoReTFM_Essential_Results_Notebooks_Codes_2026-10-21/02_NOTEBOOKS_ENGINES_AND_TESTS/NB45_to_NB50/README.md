# CoRe-TFM Research Suite v4 — 19 October 2026

## Choose a notebook

**Easiest:** `notebooks/CoReTFM_V4_ALL_IN_ONE.ipynb` (standalone). It contains everything for main two-TFM experiments, analysis, and ZIP export. Alternatively run the numbered standalone notebooks:

| Notebook | Role | Hardware |
|---|---|---|
| NB45 | Restore, validate, and inventory the existing v2 caches and frozen splits | CPU |
| NB46 | Resumable **real** TabICLv2 / explicitly versioned TabPFN-3 GPU queue | Colab T4+ |
| NB47 | Validation-selected pooling/CoRe/calibration + resampling stability | CPU |
| NB48 | Strict paired dataset-level statistics, figures and cost evidence | CPU |
| NB49 | Provenance gate, SHA-256 manifest and single results ZIP/download | CPU |
| NB50 | **Optional** TabFM model-weight license & API preflight; not a real TFM benchmark | CPU/GPU |

All main notebooks embed the same audited underlying v2 four-view engine and v4 extension. They can be run individually using the **same persisted output directory**; you don't need to upload `engine/*.py` into Colab. A source copy is included for source audit and local reuse.

## Quick start — free Google Colab

1. Open `CoReTFM_V4_ALL_IN_ONE.ipynb`, set Runtime → Change runtime type → **T4 GPU**, and use **Run all** (or run stage by stage). Google Drive is mounted by default in Colab.
2. To restore the user's last verified run, upload `CoReTFM_V2_PARTIAL_EVIDENCE (2).zip` to the Colab file pane. The setup automatically finds local `CoReTFM_V2_PARTIAL_EVIDENCE*.zip` in `/content`; alternatively set `RECOVERY_ZIP` to an absolute path. The ZIP restore only copies frozen splits and paired caches and verifies their hash manifest. It will **not overwrite** differing pre-existing files.
3. If previous results are already persisted in `MyDrive/CoRe_TFM_Resumable_v2`, no restore is necessary. By default, that Drive location is reused. The `/content/` directory is **temporary**, so mount Drive before a long queue.
4. Accept Prior Labs' official TabPFN model license and set `TABPFN_TOKEN` in **Colab Secrets** with notebook access. The token is not printed or exported. Without it, the TabPFN branch is explicitly blocked; TabICLv2 can proceed.
5. `RUN_PROFILE='BALANCED'` prioritizes five datasets and independent split seeds before expanding training caps. `RUN_GPU_QUEUE=True` executes **at most one new GPU job per pass** (`MAX_NEW_GPU_JOBS_PER_PASS=1`). Re-run the queue cell until desired coverage; no work continues invisibly in the background. Increase max jobs to 2 after successful checks if your T4 session permits it.
6. Re-run analysis and evidence cells after each batch. Set `DOWNLOAD_ZIP=True` and rerun the final export cell to download `CoReTFM_V4_RESEARCH_EVIDENCE.zip`.

## Planned scopes

- `QUICK`: 3 datasets × 2 models × 1 seed × 1 held-out split × 1 train cap = **6 configurations**.
- `BALANCED`: 5 × 2 × 3 seeds × 2 held-out splits × 3 caps = **90 configurations**.
- `FULL`: 10 × 2 × 5 seeds × 5 held-out splits × 5 caps = **1,250 configurations**. **Not intended for one free Colab session.** Model feasibility may reduce executable coverage.

The queue is resumable and prioritizes 128-example training-cap configurations across distinct datasets. It never equates an unexecuted planned cell with completed evidence. The existing three saved Car / TabICLv2 caches, for caps 64/128/256, are usable independently of the selected queue preset.

## Methodological specifics

- **Four-view inference:** direct `p(A|X)`, `p(B|X)`, conditional `p(A|B,X)`, `p(B|A,X)`. Test labels **do not** appear in conditioning inputs; all possible conditioning values are enumerated.
- **Comparators (14 policies):** arithmetic pool, geometric pool, two raw chain-rule views, hard marginal projection, soft reconciliation, pool-only selection, raw-or-pool selection, repair-or-pool selection, full candidate selection, conservative full selection, temperature-calibrated pooling, validation-optimized arithmetic pooling weight, and weighted-plus-temperature pooling. Parameters are selected from **validation labels only**.
- **Fixed-test factorial:** training caps 64/128/256 (optionally 512/1024) cross with validation-label budgets 16/32/52/103/205 on unchanged test IDs per dataset–seed–fold. The 205-label validation budget alongside cap 64 is a deliberate **decoupling** of training context from selector evidence, not a 64-total-label deployment.
- **Selection stability:** repeat random **subsets** of the *same* saved validation pool, never count those resamples as new TFM inferences or statistically independent datasets.
- **Proper scores / diagnostics:** joint NLL, joint multiclass Brier, top-1 accuracy, descriptive ECE, factorization TV, marginal distortion, per-class direct vs observed-conditioned predictive scores.
- **Statistics:** pair policies within identical dataset–model–seed–fold–cap–budget, aggregate split repeats **inside each dataset**; report per-model dataset-block bootstraps and exploratory Wilcoxon p-values with Holm correction where at least 5 independent datasets exist. Do not treat folds/seeds as independent sample units. We do **not** assert equivalence if p is non-significant.
- **Model provenance:** TabICL package `2.2.0`, checkpoint `tabicl-classifier-v2-20260212.ckpt`; TabPFN package `9.2.0`, `ModelVersion.V3`. Stable checkpoint names are recorded; **actual model weight-byte SHA-256 has not been verified**. Mixed checkpoint identities/versions within a model are not pooled by the v4 analysis.
- **Splits:** the preserved v2 engine uses independently randomized holdout splits indexed by seed and fold. These are NOT guaranteed to form disjoint fivefold cross-validation partitions; manuscript protocol language must reflect that. Splits are frozen and within-split train/val/test ID intersections must be empty.
- **Research claim:** until independent multi-dataset and both-TFM runs exist, all current comparisons are exploratory and the evidence gate stays `PARTIAL_EVIDENCE_ONLY`.

## Produced files

The persistent root `MyDrive/CoRe_TFM_Resumable_v2` contains `real_caches/*.npz` with immutable `.json` provenance, `splits/*.json`, and files under `reports/`:

`V4_JOB_COVERAGE.csv`, `V4_PER_CACHE_POLICY_RESULTS.csv`, `V4_DATASET_BLOCKS.csv`, `V4_DATASET_BLOCKED_INFERENCE.csv`, `V4_PRIMARY_COMPARISON.csv`, `V4_REAL_VALIDATION_SELECTION_STABILITY.csv`, `V4_CLASSWISE_VIEW_RELIABILITY.csv`, `V4_REAL_CONTEXT_NLL.png`, `V4_PROVENANCE.json`, `V4_ANALYSIS_STATUS.json`, `V4_EVIDENCE_GATE.json` and `CoReTFM_V4_RESEARCH_EVIDENCE.zip`.

Only V4 reports plus immutable genuine caches and frozen splits are packaged, not old V3 outputs, dataset table `.pkl`, raw tokens or model weights. The final ZIP includes a per-file SHA-256 manifest and rechecks every entry.

## Tests and honest verification limits

The v4 notebook executed end-to-end **on CPU against the uploaded three real TabICLv2 caches**. Its diagnostic analysis generated 210 method rows for 3 caches × 5 validation budgets × 14 policies, 1,080 real-validation resampling rows and 42 per-class diagnostic rows. The checksum, cache-recovery, score, incomplete-evidence, and deliberate corruption tests passed. No fresh TabICLv2, TabPFN-3 or TabFM GPU inference took place during package creation. Do not present these generated diagnostic rows as independent new model runs.

## Source and methods

- User repo: https://github.com/bnssaanirudh/core-tfm
- Official TabICL API: https://tabicl.readthedocs.io/en/latest/api.html
- Official TabPFN version + headless license: https://github.com/PriorLabs/TabPFN
- Official TabFM source/weights licensing: https://github.com/google-research/tabfm
- Scikit-learn probabilistic calibration: https://scikit-learn.org/stable/modules/calibration.html
- Demšar (2006), *Statistical Comparisons of Classifiers over Multiple Data Sets*: https://jmlr.org/papers/v7/demsar06a.html
