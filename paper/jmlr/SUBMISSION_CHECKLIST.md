# JMLR submission-readiness checklist

This checklist follows the current JMLR author information and formatting guidance. The manuscript is deliberately held as a working draft until every blocking item is complete.

## Formatting already enforced

- [x] Official `jmlr2e` style is used by the CI build; margins/geometry are not overridden.
- [x] LaTeX manuscript with a single primary `main.tex` file.
- [x] Abstract is below JMLR's 200-word submission limit (current draft: about 179 words).
- [x] Exactly five title-page keywords.
- [x] Condensed running title is below 50 characters (current: 46 characters).
- [x] Author-year citations via the JMLR/natbib bibliography style.
- [x] Hyperlinks are hidden rather than colored or boxed.
- [x] Reproducible figures are generated from committed result CSVs.
- [x] Acknowledgments/disclosures are positioned before appendices/references in the draft structure.

## Scientific blockers before submission

- [ ] Complete broader five-fold real-model benchmark on released TabPFN-3, TabICLv2, and TabFM configurations. (Car and Wine TabICLv2 pilots are complete.)
- [ ] Report model/checkpoint versions, inference budgets, hardware, and weight-license conditions.
- [ ] Re-run all adaptive choices using validation data disjoint from final test folds.
- [ ] Add paired statistical comparisons across real datasets/folds and report effect sizes.
- [ ] Recheck August 2026 and later literature immediately before submission.
- [ ] Freeze experiment seeds/configuration and create a tagged reproducibility release.

## Author/submission metadata blockers

- [ ] Replace `[Institution and postal address to be completed]` with the exact university, department/program, city, state/postal code, and country.
- [ ] Confirm the corresponding-author postal address and preferred institutional email if applicable.
- [ ] Confirm funding and competing-interest declarations.
- [ ] Add ORCID if available.
- [ ] Complete the cover letter with 3-5 non-conflicted JMLR Action Editor suggestions and 3-5 non-conflicted reviewer suggestions.
- [ ] Confirm no simultaneous submission to another journal/conference.
- [ ] Keep the final uploaded manuscript below the JMLR submission file-size limit.

## Claim guardrails

- Do not present zero post-repair incompatibility as an accuracy result.
- Do not claim invention of conditional compatibility, KL projection, IPF, or Sinkhorn scaling.
- Do not claim the first post-hoc TFM probability correction; DistPFN addresses a different problem (label shift).
- Do not claim average gains on TFMs until the released-model benchmark is completed.
- Keep coherence, calibration, proper scoring, truth distance, and dependence fidelity as distinct evaluation axes.
