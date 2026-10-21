# CoRe-TFM — Reviewed Version 1 source

- `main.tex`: fully revised Elsevier CAS-SC LaTeX source
- `references.bib`: edited 40-reference bibliography
- `Fig*.pdf` and `05_*.png`: figures used by main.tex
- `REFERENCE_CROSSCHECK_40.csv`: reference-by-reference audit, distinguishing external verification from internal checks
- `FULL_TECHNICAL_AND_STYLE_AUDIT.md`: editorial, numerical, provenance and submission review
- `STRUCTURAL_AND_NUMERIC_QA.json`: structural and quantitative validation summary
- `layout_contact_sheet.jpg`: visual page montage

Compile with `pdflatex main`, `bibtex main`, `pdflatex main`, `pdflatex main` (or a correctly configured `latexmk`). The Elsevier CAS-SC document class must be installed. Source data archives are not embedded in this package.
