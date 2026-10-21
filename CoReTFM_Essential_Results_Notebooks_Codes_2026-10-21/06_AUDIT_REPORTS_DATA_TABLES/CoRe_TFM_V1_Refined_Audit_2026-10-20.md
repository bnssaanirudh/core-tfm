# CoRe-TFM V1 — Follow-up manuscript audit (20 October 2026)

## Result
A revised 26-page first-submission manuscript has been compiled and inspected, with the source retained in LaTeX form. This is an editorial and saved-evidence audit, not a new experiment, acceptance decision, or full external review.

## Substantive corrections
- Rewrote the abstract, introductory motivation, several Related Work paragraphs, Protocol A/B comparisons, limitations, and data-availability wording. Scientific numbers were not changed.
- Removed residual workflow chronology such as “October audit,” “newer audit,” and similar phrases that made the paper read like successive notebook reports rather than one study.
- Clarified that the historical protocol uses geometric-reference projection and a penalty-based soft optimizer, whereas the fixed-test protocol uses arithmetic-reference projection and convex interpolations. Their method effects are not pooled.
- Corrected **ScoringBench** to its two listed arXiv authors, Jonas Landsgesell and Pascal Knoll. The previous bibliography contained an unsupported third name. Its related-work description now identifies its regression focus. Sources: https://arxiv.org/abs/2603.29928 .
- Distinguished the regression-specific uncertainty studies from the present paired-target classification setting.
- Replaced a procedural instruction in the article's Data Availability section with the actual status of the prepared, unpublished evidence package. No public DOI has been invented.
- Reduced repetition in the limits/reproducibility narrative and kept the key reproducibility boundaries explicit.

## Checked against the actual saved 160-configuration table
- 160 valid result rows, 10 datasets and 2 models in the combined reconstructed CSV.
- Hard repair increased test NLL in **67/80** primary 256-context configurations, while reducing prescribed marginal discrepancy in **80/80**.
- Dataset-balanced mean hard-minus-pool NLL: TabICLv2 **+0.01831**; TabPFN-3 **+0.01178**.
- Increasing context from 128 to 256 reduced arithmetic-pool NLL in **39/40 split settings per model**; mean drops 0.13332 (TabICLv2) and 0.11163 (TabPFN-3).
- **414/414** source archive manifest entries matched their stored SHA-256 values and sizes.
- These are numerical checks of stored predictions, not 160 new GPU replications.

## Citations and document quality
- **40 of 40** bibliography keys are cited; none missing or unused. All document cross-references resolve; labels are unique.
- **13 specific external reference records** were inspected in this follow-up pass, including one corrected author-list error. The table `REFERENCE_AUDIT_40_UPDATED.csv` distinguishes these from prior author/year/internal checks. Not all 40 papers were reread in full.
- The updated PDF has **26 pages, 10 figures and 17 tables**. Text spans fall within page bounds; figures 9 and 10 appear on page 19, before the references on page 25. This is a programmatic page-boundary/structure check plus selected visual inspection, not a guarantee that every journal production system will render identically.

## Limits to preserve in submission
- No persistent public DOI has been assigned to the prepared research-data deposit. Release remains pending.
- Checkpoint fingerprints were recorded for six targeted fresh replications; they do not independently certify which weight bytes all 160 original GPU jobs loaded.
- The MIC raw-download checksum was reported by the standalone verification process; its ARFF bytes were not included in the verification ZIP, so manuscript-side independent rehashing was not possible.
- Historical raw-data transformation identity is not fully certified for both protocols; the Diamonds construction particularly needs an end-to-end schema review if claiming strict raw-to-cache replay.
- Dataset-level effects are exploratory and the Holm-adjusted hard-repair tests do not meet a 0.05 threshold. The guarded repair policy's zero activations do not demonstrate repair benefits.
- Final publication requirements, coauthor declarations and rights-cleared data deposit should be checked before submission to Knowledge-Based Systems.

## Language-editing method
Prose was edited manually for specificity, shorter transitions and a natural academic voice. Technical terms, equations, results and limitations were protected from indiscriminate paraphrasing. AI-detector scores were not used as a target or proof of authorship.
