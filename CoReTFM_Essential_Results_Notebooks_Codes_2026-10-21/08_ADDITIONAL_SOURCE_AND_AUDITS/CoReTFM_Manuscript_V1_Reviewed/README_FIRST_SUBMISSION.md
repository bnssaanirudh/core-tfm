# CoRe-TFM — first-submission manuscript

Open `CoRe_TFM_Manuscript_V1.pdf` for the complete article, including appendices. The editable project is in `manuscript/`.

To compile with an Elsevier CAS single-column TeX distribution installed:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode main.tex
```

The project uses BibTeX and the `cas-model2-names` bibliography style. Source files include `main.tex`, `references.bib`, original and new study figures. The separate `REFERENCE_CROSSCHECK_40.csv` and `MANUSCRIPT_AND_REFERENCE_AUDIT.md` are editorial records, not part of the submitted article.

The manuscript is an integrated first-submission version in presentation, while accurately distinguishing the two separate experimental protocols and their respective evidence. Data availability still needs a real permanent evidence DOI before submission.
