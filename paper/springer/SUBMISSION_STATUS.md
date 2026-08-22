# Submission status

## Ready in the draft package
- Springer Nature `sn-jnl` manuscript source structure.
- 150-250 word abstract and six keywords.
- Author-year citations and working bibliography.
- Decimal section hierarchy.
- Formal problem statement and propositions.
- Hard, Soft and Selective CoRe methods.
- Exact-view perturbation mechanism experiment (6 regimes x 20 seeds).
- Surrogate DGP robustness sweeps.
- Paired statistical protocol with Holm correction.
- Reproducibility/data protocol and declarations sections.
- Required Machine Learning Contribution Information Sheet.

## Submission blockers
1. Complete the broader released-TFM cross-validation benchmark; the completed five-fold Car and Wine TabICLv2 pilots are two cells, not support for a general TFM performance claim.
2. Fill exact university/department/postal affiliation.
3. Confirm funding, competing interests, final author list/contributions and GenAI disclosure wording.
4. Recheck all reference metadata and novelty immediately before submission.
5. Build with the official Springer Nature LaTeX v3.1 (December 2024) package and archive the exact submitted source/PDF.
6. Create an immutable code/data release (for example Zenodo) and add DOI if available.

## Real-model status
Five-fold leakage-free TabICLv2 2.1.1 results are archived for Car and Wine. Car has factorization TV `0.06956 ± 0.00475`; Selective CoRe chooses raw $B\rightarrow A$ in 4/5 folds and reaches `1.11691 ± 0.00786` joint NLL. Wine has factorization TV `0.02450 ± 0.00200`; fixed arithmetic pooling is best (`0.71678 ± 0.01085`) and Selective CoRe chooses arithmetic pooling in 2/5 folds and Soft CoRe in 3/5 (`0.71797 ± 0.01237`). Wine uses a source-equivalent canonical-UCI reconstruction with archived hashes. The earlier Credit OpenML HTTP 504 is infrastructure-only and is not a model result.
