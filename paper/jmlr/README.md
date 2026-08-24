# JMLR-formatted working draft

Primary top-tier target format for the CoRe-TFM manuscript.

- `main.tex` uses the official JMLR `jmlr2e` style in preprint mode.
- `references.bib` contains the working bibliography.
- `Fig1_exact_view_regret.pdf`, `Fig2_selective_task_gain.pdf`,
  `Fig3_validation_size.pdf`, and `Fig4_real_dataset_effects.pdf` are generated
  from the committed final evidence package.
- `CoRe_TFM_JMLR_Draft.pdf` is generated in GitHub Actions from the official `jmlr2e.sty` fetched from the JMLR style repository.

This is **not submission-ready**. The bounded 10-dataset, two-TFM benchmark is
complete and reflected in `main.tex`, including its negative primary result.
Independent sampling-seed and context-size sensitivity, rare-class robustness,
exact institutional address, disclosures, and final author metadata remain open.

JMLR submission formatting is stricter than generic LaTeX: do not alter margins or the official style file. The accepted-camera-ready metadata (`jmlrheading`, editor, dates, paper ID) is intentionally not populated in the preprint working draft.

# JMLR CI downloads the official style and writes the compiled draft PDF.
