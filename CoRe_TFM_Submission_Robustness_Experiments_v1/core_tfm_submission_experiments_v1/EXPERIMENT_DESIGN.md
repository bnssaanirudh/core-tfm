# Experiment Design

## E0 — Frozen evidence audit (no new inference)

### Purpose
Verify that the original completed paper evidence has not changed before running
new experiments.

### Input
`results/q1_fast_complete_256_v1/`

### Command
The finalizer invokes:
`experiments/audit_evidence_package.py`

### Pass criteria
- no missing required files
- 1,200 fold/method rows
- 8 methods per fold
- no checksum mismatch
- canonical primary result remains unchanged

This is an integrity check, **not a replication**.

---

## E1 — Independent constrained-sampling seed robustness

### Hypothesis
The observed Selective-CoRe vs arithmetic effect should not be interpreted as
stable across sampling realizations until it has been repeated with independent
seeds.

### Minimal preset
Seeds: `11, 23, 42`

### Full preset
Seeds: `11, 23, 42, 71, 101`

### Fixed design
- train limit: 256
- 5 outer folds
- validation mechanism inherited from the existing frozen Q1 notebook
- test limit: 128
- primary TFMs: TabICLv2, TabPFN-3
- CatBoost: descriptive boundary model
- no test-set tuning

### Primary analysis
For each seed:
1. average folds within `(dataset, model)`;
2. average the two primary TFMs within dataset;
3. compute `Selective - Arithmetic` NLL per dataset;
4. report seed-specific dataset-blocked mean and sign;
5. report distribution of seed-level effects.

### Guardrail
Do not treat all folds/cells as independent confirmatory observations.

---

## E2 — Training-context sensitivity

### Purpose
Test whether the conclusion is an artifact of the bounded 256-row context.

### Minimal preset
- train sizes: 256, 512, 1024
- seeds: 23, 42
- 5 folds
- test limit: 128

### Full preset
- train sizes: 64, 128, 256, 512, 1024
- seeds: 23, 42, 71
- 5 folds
- test limit: 128

### Outcome
Dataset-blocked `Selective - Arithmetic` NLL by context size.

### Operational rule
If a model/checkpoint cannot support a requested context on the target hardware,
record that cell as operationally unsupported. Do not silently replace the model,
lower the context, or reuse another result.

---

## E3 — Rare-class / low-support sensitivity

### Audited low-support datasets
- Customer: minimum target support 3
- Marketing: 2
- Nursery: 2

### Thresholds
`1, 2, 5, 10`

### Analyses
1. exclusion sensitivity at each minimum-support threshold;
2. report dataset-level effect with support metadata;
3. compare the conclusion with and without low-support datasets.

### Important
This is a sensitivity analysis. Excluding a dataset after looking at its result
is not allowed. Thresholds are frozen in advance.

---

## E4 — Safe / complexity-aware selection

### Scientific question
The original selector has a non-trivial candidate family under a small validation
budget. Is the loss relative to arithmetic mainly **selection regret** rather
than lack of candidate headroom?

### Candidate families
- `fallback`: arithmetic
- `raw_pool`: J1, J2, arithmetic, geometric
- `repair_small`: arithmetic, geometric, soft
- `full`: J1, J2, arithmetic, geometric, hard/MPR, soft

### Validation sizes
`26, 52, 104, 208`

### Beta grid
`0.25, 0.5, 1.0, 2.0`

### Required records
- all validation candidate scores
- selected policy
- arithmetic loss
- oracle-within-family loss (analysis only)
- available opportunity = arithmetic - oracle
- selection regret = selected - oracle
- selected - arithmetic

### Guardrails
- no test-set tuning
- oracle is analysis-only
- arithmetic remains the fallback

---

## E5 — View-reliability diagnostics

### Save per dataset/model/fold
- direct marginal NLL for A and B
- conditional NLL for A|B and B|A
- corresponding Brier/ECE where available
- factorization TV
- marginalization defects
- selected-minus-arithmetic effect

### Analysis
Use dataset/model/fold records descriptively and dataset-blocked aggregation for
paper-level inference. Correlation does not establish causation.

---

## E6 — Controlled Safe-Selective experiment

This is CPU-side and uses controlled known-truth tasks. It is intentionally
separate from the real TFM matrix.

### Frozen bundle defaults
- 100 tasks
- n = 2200
- beta = 1
- delta = 0.05
- probability floor = 1e-12

### Purpose
Demonstrate selection/fallback mechanics under known truth without using test
labels to tune the real-data selector.

---

# Completion criteria

A robustness package is paper-admissible only after:
- all requested variants complete, or unsupported variants have explicit failure logs;
- CPU controlled experiment completes;
- aggregate analysis exists;
- rare-class analysis exists;
- selection-ablation and validation-sensitivity outputs exist;
- view-reliability archive exists;
- completion gate passes;
- environment metadata and manifests exist;
- final evidence manifest + SHA-256 checksums are generated.

No value from a partial directory is to be described as a completed paper result.
