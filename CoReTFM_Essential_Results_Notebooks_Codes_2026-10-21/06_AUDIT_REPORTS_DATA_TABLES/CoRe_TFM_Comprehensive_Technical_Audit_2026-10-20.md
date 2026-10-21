# CoRe-TFM — comprehensive manuscript and evidence audit

**Review date:** 20 October 2026  
**Deliverable:** `CoRe_TFM_Enhanced_2026-10-20.pdf` (Elsevier CAS single-column author-review draft)  
**Basis:** 12 October original Elsevier draft; historical notebook evidence as documented in the original text; complete October 19–20 combined 160-cache artifact; final v7 independent-audit reports; a new CPU-only geometric-vs-arithmetic projection-reference ablation.

## Executive judgment

The central empirical question is clear and now supported by ten datasets, two real tabular foundation models, matched validation/test IDs and an independently audited 160-cache dataset. However, **the package should not be described as submission-cleared or as a proof that the full CoRe framework improves forecasting**. The important new finding is that enforcing the model's direct-marginal constraints can make its held-out joint forecast worse. Original and newer protocol variants, projection-reference choices, and the scope of statistical inference must be presented precisely.

## 1. Independently verified evidence checks

| Item | Check | Outcome |
|---|---|---|
| Combined evidence container | ZIP CRC test and file inventory | PASS |
| Combined SHA-256 manifest | Every listed path, exact bytes and SHA-256 | **414 / 414 PASS** |
| Real-model prediction caches | Saved NPZ SHA-256 and JSON sidecars | **160 / 160 PASS** |
| Real-model output domains | Probability finiteness and class-axis normalization for all four query views | **160 / 160 PASS** |
| Label/data partitioning | No train, validation or test row-ID intersection in any cache | PASS |
| Frozen splits | 40 saved specifications from combined archive | Present; separately verified in v7 audit |
| Training context | 128 and 256 rows, with 205 reserved validation and 128 reserved test rows | MATCHES SAVED CACHE METADATA |
| TFMs | TabICLv2 (2.2.0) and TabPFN package (9.2.0, V3 interface) | MATCHES METADATA; actual model weights not authenticated |
| Primary scores | 160-cache audit and new summary recomputed from `03_CONFIGURATION_METHODS.csv` | PASS |
| PDF build | 26 pages, citations and cross-references resolved, 182 internal/external PDF links | PASS |
| Physical PDF clipping | PyMuPDF block-boundary check | **0** outside-page text blocks |
| Historic original five-seed study | The existing 12 Oct narrative and statistics preserved, **not** independently rerun in this audit | NOT REVERIFIED |

A valid SHA-256 of predictions establishes **artifact integrity**, not that a particular historically loaded checkpoint or original OpenML download produced them.

## 2. Material scientific corrections completed

1. **Two different hard-projection definitions were previously conflated.** The original optimization projected a normalized **geometric** joint onto the direct marginals. The newer v2–v7 cache audit projected an **arithmetic** joint. The manuscript now defines a parameterized information projection and states which reference distribution is used in each protocol.
2. **Soft CoRe is not the same thing as a convex blend.** The original penalty-based KL optimizer is distinct from the newer 25%, 50% and 75% interpolations between an arithmetic pool and its hard projection. The methods and labels are now distinguished.
3. **Compatibility does not certify predictive improvement.** A strict-propriety counterexample shows that if the unrepaired pool equals the true joint, any distinct projected joint has increased expected NLL equal to a positive KL divergence. When the projection equals truth, the sign reverses. The argument is stated as an elementary consequence of proper scoring, not claimed as a novel theorem.
4. **Self-consistency vs ground-truth calibration distinguished.** Marginal TV measures agreement with the same TFM's direct predictions; its reduction is expected by construction and does not demonstrate increased truthfulness.
5. **Old and new benchmark variants are not pooled.** The historical five-fold/five-seed, up-to-48-candidate study and the newer four-randomized-split/eight-base-candidate study remain separate, including their different package versions and Diamonds feature treatment.
6. **Evidence sample sizes clarified.** The newer experiment has 160 model–dataset–split–context caches and 1,920 derived policy rows, but only **ten** dataset-level inferential units. Four randomized split settings are correlated within datasets, not four independent datasets.
7. **Training-size label budget clarified.** A 128-row training context does **not** mean 128 total task labels: 205 additional validation labels and 128 held-out test labels were reserved. The design isolates context size under fixed held-out samples.
8. **Selection and multiplicity claims tightened.** Raw-or-pool selection is descriptively competitive but not established as significantly superior. The revised document preserves the exploratory Holm-adjusted p-values and explicitly acknowledges that historical and post-hoc comparisons multiply the research degrees of freedom.
9. **Guarded repair non-activation stated precisely.** Zero activations mean the recorded policy behaved identically to its calibration fallback; this is not evidence that it identifies favorable repair cases or protects from shift.
10. **Runtime and data-feature limitations expanded.** Extra target-conditional estimators and hypothetical conditioned feature values impose additional cost and out-of-distribution risks; class-coverage frontloading is nonrandom; the cached timing does not support steady-state latency claims.

## 3. Core 160-cache primary results (256 training, first 52 validation labels)

| Method | TabICLv2 mean NLL | TabPFN-3 mean NLL |
|---|---:|---:|
| Arithmetic pooling | 1.34000 | 1.34147 |
| Raw-or-pool selector | 1.33919 | 1.33926 |
| Restricted eight-candidate selector | 1.33994 | 1.33944 |
| Temperature-fitted pooling | 1.35350 | 1.35492 |
| Arithmetic-reference hard repair | 1.35831 | 1.35325 |

Average effects are computed over the **ten dataset-averaged** values per model. Negative differences indicate improved NLL; the selector advantage is tiny. The manuscript now also shows the complete Brier, top-1 accuracy and ECE table. ECE is treated as descriptive and dependent on binning.

### Primary structural ablation

- Hard arithmetic-reference repair reduces marginal discrepancy in **80/80** model–dataset–split settings, as expected from the operator's constraints.
- Held-out NLL worsens in **67/80** settings (TabICLv2 **36/40**, TabPFN-3 **31/40**).
- Across ten dataset averages, hard repair harms **9/10** datasets for each model.
- TabICLv2 mean change **+0.01831 NLL**, 95% dataset-bootstrap interval **[+0.00893, +0.02832]**, exploratory unadjusted Wilcoxon p **0.00977**, Holm p **0.06836**.
- TabPFN-3 mean change **+0.01178 NLL**, interval **[+0.00397, +0.01954]**, unadjusted p **0.01953**, Holm p **0.13672**.
- The intervals are descriptive dataset-resampling summaries; they do not cancel the exploratory multiple-comparison issue.

### Newly derived fixed-test context sensitivity

For arithmetic pooling, increasing the training context from 128 to 256 while keeping validation and test IDs unchanged reduces NLL by **0.13332** (TabICLv2) and **0.11163** (TabPFN-3) on average. The context increase improves NLL in **39/40** paired randomized splits and **10/10** dataset averages for each model. These gains should not be attributed to CoRe repair.

### Newly derived projection-reference bridge (not new GPU training)

We independently reconstructed geometric-reference projection directly from the original probability arrays and compared it with the arithmetic-reference operator for all saved configurations. At context 256:

| Comparison | TabICLv2 delta NLL | TabPFN-3 delta NLL |
|---|---:|---:|
| Hard(arithmetic reference) minus arithmetic pool | +0.01831 | +0.01178 |
| Hard(geometric reference) minus arithmetic pool | **+0.01800** | **+0.01126** |
| Hard(geometric reference) minus geometric pool | +0.01836 | +0.01192 |
| Hard(geometric reference) minus Hard(arithmetic reference) | -0.00031 | -0.00052 |

Geometric-reference repair worsens arithmetic-pool NLL on **9/10 datasets** for both models; both references attain near-zero direct-marginal discrepancy. **This additional intervention was planned after inspection of earlier results, is post-hoc, and does not establish the correctness of every possible CoRe variant.** Scripts and paired CSVs are in the LaTeX project.

## 4. Bibliography and manuscript QA

- Preserved **40 distinct bibliography keys** and the parent manuscript's citation coverage.
- Repaired demonstrably malformed title strings where proceeding names, page ranges, or arXiv identifiers had been copied into the paper title.
- Distinguished journal articles, conference proceedings, repository items and preprints in the editable `references.bib`; provided a self-contained `references_manual.tex` for compilation without external BibTeX binaries.
- Independently checked the identity/details of high-priority modern works against externally accessible records: Qu et al., TabICLv2 (PMLR ICML 2026), TabPFN-3 technical report (official Prior Labs citation and arXiv 2605.13986), Klötergens et al. arXiv 2608.06004, ScoringBench arXiv 2603.29928, Costa et al. arXiv 2605.28554, Johnson et al. arXiv 2606.01427, and Tanna et al. arXiv 2605.18696.
- **Not all 40 publisher records were individually reauthenticated**. A final author-run DOI/reference manager sweep remains required; some citations are preprints or workshop reports rather than journal publications.
- Verified figure inclusion, captions, table labels, resolved citations, page order and PDF bounds. The sole `Overfull \\hbox` compile warning is in Elsevier CAS front-matter (line 60); the title page was inspected and no visual clipping was observed. Three minor float-position warnings arise from the CAS template and have no observed clipping.

## 5. Highest-priority unresolved blockers

### Blocking a strong reproducibility claim

1. **Actual loaded checkpoint bytes**: historical TabICLv2 and TabPFN-3 weights were not hashed at execution time. Current metadata names model interfaces/checkpoint labels but records no authenticated SHA-256 of the bytes used for these predictions.
2. **Raw data provenance**: the exact OpenML dataset version/revision, original raw byte checksum and unambiguous paired-label/feature transformation should be deposited. Five processed dataset snapshots are hashed; these are **not** equivalent to ten original raw downloads.
3. **Clean pinned independent replication**: rerun a small selected group of cross-model tasks (e.g. Car, Nursery, Diamonds with both context sizes) in a pinned environment, save original checkpoint paths and hashes, and match previously frozen validation/test IDs.
4. **Immutable evidence deposit**: archive both S1 (historical) and S2 (new 160-cache and v7 audit), with a stable DOI, documented rights/license terms, complete reproducibility instructions, and public source commit. Current evidence ZIP filenames are local artifact names only.

### Blocking final editorial sign-off

5. All three authors must verify their names, corresponding-author contact, individual CRediT contribution assignments, funding declaration, conflict statement and publisher-required AI declaration; assistant-generated author roles cannot substitute for author confirmation.
6. Complete reference-by-reference DOI/publisher validation. The corrected entries are a substantial improvement over PDF-transcribed titles, but are not yet certified as a perfect journal bibliography.
7. Select a specific target Elsevier journal and validate its current submission guidelines, template/class options, manuscript length, ethical statements, data availability and supplementary policies. CAS single-column alone does not establish journal compliance.

### Scientifically valuable but not a barrier to an author-review draft

8. A genuinely predictive validation-time repair gate should be evaluated on independent datasets if the method is to be marketed as a validated improvement: the existing gate never activated.
9. Add an out-of-distribution/reliability-shift benchmark, calibration-decomposition analysis, third TFM and steady-state fit/inference cost/memory results, if the target venue requires stronger algorithmic novelty or deployment claims.
10. The original historical 17-run results were retained faithfully from the previous manuscript, but an independent raw-input audit of their entire notebook lineage was not performed in this revision. Do not imply the October 160-cache reanalysis verifies their historic source code or model weights.

## 6. Recommended publication framing

**Title:** *Coherence Is Not Accuracy: Evaluating Reconciliation, Calibration, and Reliability-Aware View Selection in Tabular Foundation Models*.

**Claim supported:** In the audited paired-label classification setting, enforcing direct-marginal agreement is not sufficient to improve held-out proper scores. In this dataset collection, arithmetic- and geometric-reference projections often harmed NLL even as structural mismatch was removed. A simple raw-view-or-pool selector was competitive without a statistically established advantage over arithmetic pooling. Dataset differences (especially Diamonds) limit any universal "repair always harms" assertion.

**Claim not supported:** CoRe globally improves predictive accuracy; the guarded gate has learned to identify beneficial repairs; 160 configurations are 160 statistically independent datasets; the full experiment is checkpoint/data-byte reproducible; the original historical optimization and newer proxy implementations are interchangeable.

## Source and reproducibility files

- `main.tex` — revised manuscript.
- `references.bib` — editable structured citation metadata.
- `references_manual.tex` — portable bibliography used in compiled PDF.
- `RESULTS_RECOMPUTED_V2.json` — secondary scores and matched-context statistics.
- `BRIDGE_PROJECTION_CONFIGURATION_SCORES.csv` — 160 saved-cache arithmetic/geometric projection-reference comparisons.
- `BRIDGE_PROJECTION_DATASET_BLOCKS.csv`, `BRIDGE_PROJECTION_SUMMARY.csv` — dataset-level bridge evidence.
- `bridge_projection.py` — independently reproducible bridge analysis; run against the user's verified combined evidence ZIP (no GPU).
- `NEW_EVIDENCE_INTEGRITY_AUDIT.json` — complete 414-entry + 160-cache checksum/probability/partition audit.
- 8 historical vector PDF figures and 5 newer PNG figures available in the source package.

The review is a comprehensive artifact, methodology and documentary audit of the **materials actually available**. It is **not** a raw-data reproduction of all prior notebooks, a prospective validation, an institutional ethics review, a plagiarism similarity scan or a guarantee of journal acceptance.
