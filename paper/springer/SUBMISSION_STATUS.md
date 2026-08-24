# Submission status

## Ready in the draft package
- Springer Nature `sn-jnl` manuscript source structure.
- 150-250 word abstract and six keywords.
- Author-year citations and working bibliography.
- Decimal section hierarchy.
- Formal problem statement and propositions.
- Hard, Soft and Selective CoRe methods.
- Exact-view perturbation mechanism experiment (6 regimes x 50 seeds).
- Heterogeneous 100-task controlled selection benchmark and 50-task validation-size study.
- Complete bounded real-model matrix (10 datasets x 3 models x 5 folds), with two TFMs primary and CatBoost descriptive.
- Dataset-blocked inference, selection-family ablation, validation-fraction sensitivity, and checksum-verified evidence archive.
- Surrogate DGP robustness sweeps.
- Paired statistical protocol with Holm correction.
- Reproducibility/data protocol and declarations sections.
- Required Machine Learning Contribution Information Sheet.
- Fresh 16-page PDF compiled with the official Springer Nature v3.1 package;
  citations and cross-references resolve, and every page has been visually checked.

## Submission blockers
1. Complete independent sampling-seed, context-size, and rare-class robustness; preferably add a third released TFM.
2. Fill exact university/department/postal affiliation.
3. Confirm funding, competing interests, final author list/contributions and GenAI disclosure wording.
4. Recheck all reference metadata and novelty immediately before submission.
5. Create an immutable code/data release (for example Zenodo) and add DOI if available.

## Real-model status
The bounded matrix is complete: 150 fold tasks and 1,200 method rows. Across ten
dataset units after averaging TabICLv2 and TabPFN-3, Selective CoRe is worse than
arithmetic pooling by `+0.001831` NLL (95% CI `[+0.000548,+0.003301]`, 2/10 wins,
`p=0.02734`). It reduces marginal distortion by `0.003855` but worsens Brier score
by `0.000646`. Effects reverse by model: `+0.006694` on TabICLv2 and `-0.003031`
on TabPFN-3. CatBoost is descriptive and is not treated as a TFM replacement.
