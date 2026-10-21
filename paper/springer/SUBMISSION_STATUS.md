# Submission status

## Ready in the draft package
- Completed S2 evidence release: 10 datasets x 2 TFMs x 4 held-out split settings x 2 training caps (160 configurations), published with Zenodo DOI [10.5281/zenodo.23507196](https://doi.org/10.5281/zenodo.23507196).
- Six targeted clean reruns cover three datasets x two TFMs; these do not constitute six full-matrix replications.
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
- A pre-S2 16-page PDF was compiled with the official Springer Nature v3.1 package;
  regenerate and review it after synchronizing the manuscript with S2.

## Submission blockers
1. The six clean reruns are targeted checks, not full-matrix sampling-seed or context-size sensitivity; checkpoint-weight byte identity is not verified in the S2 audit.
2. Complete any remaining manuscript analyses and make the manuscript's claims match the S2 evidence and its dataset-level inferential units.
3. Fill exact university/department/postal affiliation.
4. Confirm funding, competing interests, final author list/contributions and GenAI disclosure wording.
5. Recheck all reference metadata and novelty immediately before submission.

## Real-model status
The current S2 primary matrix contains 160 configurations (10 datasets, two TFMs,
four held-out split settings, and two training caps), with 320 cache and metadata
files and 40 frozen split specifications. The matrix audit reports 160 valid
configurations and zero audit errors. Its split settings are repeated tests within
datasets; the dataset is the inferential unit. Six clean reruns verify three
datasets with both TFMs in a separate environment. The older 150-task, five-fold
statistics above are superseded by the S2 evidence package and should not be used
as current results.
