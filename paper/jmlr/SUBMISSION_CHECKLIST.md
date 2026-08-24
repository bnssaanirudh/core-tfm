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
- [x] Reproducible figures are generated from the committed final evidence package.
- [x] Acknowledgments/disclosures are positioned before appendices/references in the draft structure.
- [x] Fresh 18-page PDF compiled from the synchronized source with all citations and cross-references resolved; every page was visually checked.

## Scientific blockers before submission

- [x] Complete the bounded five-fold, ten-dataset benchmark on TabICLv2 and TabPFN-3, with CatBoost clearly separated as a non-TFM boundary baseline.
- [x] Report package versions, bounded inference budgets, hardware, and weight-license conditions.
- [x] Run all adaptive choices using validation data disjoint from final test folds.
- [x] Add dataset-blocked paired comparisons and effect sizes for NLL, Brier, ECE, and marginal distortion.
- [ ] Repeat constrained sampling with at least three independent seeds.
- [ ] Add 256/512/1,024/full-feasible training-context sensitivity on representative datasets.
- [ ] Resolve Marketing and Nursery rare-class robustness, including an exclusion sensitivity.
- [ ] Add a third released TFM or narrow all cross-TFM claims explicitly to the two evaluated models.
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
- Do not claim average Selective-CoRe gains on TFMs; the completed primary result favors arithmetic pooling.
- Treat the 30-cell analysis as descriptive. The confirmatory unit is the dataset after averaging the two TFMs.
- Keep coherence, calibration, proper scoring, truth distance, and dependence fidelity as distinct evaluation axes.
