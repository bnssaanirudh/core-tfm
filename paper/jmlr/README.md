# JMLR-formatted working draft

Primary top-tier target format for the CoRe-TFM manuscript.

- `main.tex` uses the official JMLR `jmlr2e` style in preprint mode.
- `references.bib` contains the working bibliography.
- The existing manuscript figures and compiled PDF predate the S2 evidence and
  must be regenerated after `main.tex` and the result tables are synchronized.
- `CoRe_TFM_JMLR_Draft.pdf` is generated in GitHub Actions from the official `jmlr2e.sty` fetched from the JMLR style repository.

The current S2 evidence release is complete: 160 configurations across ten
datasets, two TFMs, four held-out split settings, and two training caps; six
targeted clean reruns; and a published Zenodo dataset
([10.5281/zenodo.23507196](https://doi.org/10.5281/zenodo.23507196)). The earlier
five-fold benchmark and its readiness wording are historical. The manuscript and
venue-specific submission checklist must stay synchronized with the S2 evidence;
the clean reruns are not full-matrix seed replications, and checkpoint-weight byte
identity remains unverified in the released audit.

JMLR submission formatting is stricter than generic LaTeX: do not alter margins or the official style file. The accepted-camera-ready metadata (`jmlrheading`, editor, dates, paper ID) is intentionally not populated in the preprint working draft.

# JMLR CI downloads the official style and writes the compiled draft PDF.
