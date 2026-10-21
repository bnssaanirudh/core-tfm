# CoRe-TFM — Comprehensive manuscript, citation and evidence audit

**Editorial build:** 20 October 2026  
**Basis:** the MIC-verified first-submission source, original 160-cache combined evidence archive, six clean GPU replication files, and standalone MIC verification report.  
**Scope:** checked document structure, all citation keys and bibliography entries for internal consistency, selected high-priority source metadata against public records, principal numerical outcomes against stored results, PDF rendering, and prose in major narrative sections.

## Editorial decision

The manuscript is in suitable condition for final coauthor inspection. The central findings are supported by the saved data, but **the manuscript cannot truthfully claim full historical weight-byte authentication, historical raw-preprocessing byte equivalence, confirmatory family-wise statistical significance, or a public supplementary DOI**. These limits remain explicit. This review does not substitute for the target journal's editorial and ethics checks.

## Changes made

1. Rewrote the abstract to place the research question, principal protocol, observed intervention effect, and limitations in a single coherent account. Reduced generic opening claims and distinguished algebraic consistency from predictive risk.
2. Reorganized the introduction around the actual two-outcome query setting, the two chain-rule joints, and the practical decision to project or retain a pool. Rewrote the contributions to avoid implying novelty in iterative proportional fitting.
3. Reworked Related Work to connect cited studies to the precise question being evaluated, rather than presenting a chronological or generic survey. Retained all 40 original citation keys.
4. Rewrote the Discussion to explain *why* a projection may fail under inaccurate marginals, what the Diamonds exception shows, why a non-activating gate proves no deployment benefit, and why Protocol A and Protocol B must not be pooled.
5. Rewrote the Conclusion in measured language, keeping all principal numbers unchanged and acknowledging the adjusted exploratory tests.
6. Corrected instances of overconfident “confirmatory” terminology in the methodological language; explicit exploratory/multiplicity caveats were retained.
7. Updated the TabArena reference from an arXiv-only entry to its documented NeurIPS 2025 proceedings record and added verified pages/URL for Dirichlet calibration (NeurIPS 2019).
8. Changed figure placement so the two new direct-ablation illustrations appear with the results (page 19), not as float-only pages *after* the end declarations. The two diagnostic panels are now displayed side by side. Original vector methodological figures are retained.
9. Preserved the separate original geometric-reference KL projection versus newer arithmetic-reference projection, and preserved the distinction between the penalty-based historical Soft CoRe and the newer interpolation variants.
10. Kept author names, corresponding-author details, author declarations, and the AI-assisted-language-editing disclosure; did not invent roles, grants, conflicts or consent.

## Evidence rechecks performed locally

| Check | Outcome |
|---|---|
| Combined evidence ZIP integrity | 414/414 SHA-256 manifest entries and sizes match |
| Independently saved model configurations | 160 total, 80 at the primary 256-row context |
| Hard repair marginal discrepancy | Reduced in 80/80 primary cases |
| Hard repair test NLL | Increased in 67/80 primary cases |
| Dataset-balanced mean hard minus arithmetic NLL | TabICLv2 +0.018310; TabPFN-3 +0.011781 |
| Six clean replication `.npz` files | Exact byte-for-byte matches to their archived counterparts |
| Standalone MIC checksum JSON | Official/downloaded MD5 fields match; the targets `LET_IS` and `ASP_S_n` are recorded |
| Historical original loaded-weight hashes | Not established for all 160 runs |
| Raw OpenML versions and preprocessing equivalence | Download-check reports available, historical transformation equivalence not fully certified |

The standalone MIC ZIP contains the JSON verification report but **not** the downloaded ARFF. Its MD5 match was reported by the Colab session; the ARFF bytes were not independently rehashed inside this audit. These facts must be described separately.

## Scientific and statistical claims retained

- The two experimental protocols use distinct model package versions, candidate families and selection budgets. Their rows cannot be treated as one pooled inferential sample.
- Repeated split settings are correlated within a dataset; the dataset ($n=10$) is the unit for dataset-blocked inference.
- The 160 model configurations are *not* 160 independent datasets. The six clean replications are checks of existing configurations, not six new data points.
- The later 205-example validation pool remains separate from the 128/256-example training context and 128-example fixed test set. “256-row” is therefore not a total-label-budget claim.
- The Holm-adjusted exploratory results do not support claiming universal predictive harm or statistically proven superiority of the raw-or-pool selector.
- The guarded repair rule did not activate: it cannot be credited with identifying favorable interventions.
- The Diamonds mean improvement is heterogeneous across splits and cannot be elevated into a validated deployment gate.

## Bibliographic checks and limits

All **40** bibliography keys are cited in the text. No undefined citation, unused bibliography entry, duplicate `\label`, or missing cross-reference was found after rebuilding the manuscript. Bibliography output was regenerated with Elsevier's author-year style.

**16 selected high-priority records were compared against public sources in this editorial pass**, including TabArena, TabICLv2, TabPFN-3, scoring-rule/uncertainty preprints, Muré (2019), Gneiting et al. (2007), Kull et al. (2019), the primary relevant KBS papers, and the AISTATS conditional-compatibility paper. The separate `REFERENCE_CROSSCHECK_40.csv` identifies exactly which items were externally checked and provides source URLs. The other 24 entries passed the internal citation/BibTeX cross-check, but were not each freshly authenticated against publisher metadata during this pass. Prior versions' checks are not misrepresented as new independent verification.

**One metadata watch item:** the ScoringBench project citation lists a three-author record including Tizian Wenzel, whereas the indexed arXiv abstract may list only two authors. The bibliography currently follows the project’s posted citation; the authors should confirm the specific arXiv manuscript version before journal submission. This is not a technical source falsification finding, but warrants a final check.

## Writing-quality and AI-style review

A writing-quality diagnosis and a humanizer workbench were invoked on representative manuscript passages. Those services opened interactive review panels; no automatic full-document humanization or trustworthy “AI percentage” was obtained. The actual LaTeX edits were written and checked against the evidence, including reworking repetitive transitions, replacing generic significance language with quantitative interpretations, varying sentence patterns without adding rhetorical flourishes, and removing inflated novelty claims. Automated AI-text classifiers cannot reliably establish authorship, and no detector-pass percentage is claimed.

The AI-assisted-editing declaration remains in the manuscript. All scientific responsibility stays with the named authors.

## PDF technical QA

- Successful final LaTeX/BibTeX rebuild with **27 pages**, **40 bibliography entries**, **10 figures**, and **17 tables**.
- No unresolved citation or reference markers, duplicate labels, or out-of-page text spans found.
- Results Figures 9 and 10 placed together on page 19; no orphaned figure-only pages follow the declarations.
- Figure and table content remains extractable and visually legible on review. The Elsevier CAS title-front-matter hbox diagnostic may still appear in TeX logs, but no title-page clipping was observed in the rendered PDF.

## Still needed for submission

1. Publish the rights-cleared evidence package through the accepted repository and insert the resulting persistent DOI or journal-approved data-access link. **No DOI has been invented.**
2. Reconfirm historical preprocessing scripts and feature mappings against the actual frozen raw versions if the paper claims end-to-end exact raw-to-cache reproduction. The existing paragraph correctly limits that claim.
3. Confirm the ScoringBench author-list metadata and, ideally, the remaining 24 references against authoritative sources before final submission.
4. Run the target journal's final author declaration, graphical-abstract/highlights and format checklist; obtain a last proofread from all coauthors.

**Status:** revised master V1 for author review, not falsely labeled as formally accepted or publicly deposited.
