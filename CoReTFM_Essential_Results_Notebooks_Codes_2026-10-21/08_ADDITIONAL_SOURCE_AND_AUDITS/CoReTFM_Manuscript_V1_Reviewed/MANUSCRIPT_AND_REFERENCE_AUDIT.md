# CoRe-TFM — Version 1 Manuscript Editorial and Citation Audit

**Date:** 20 October 2026  
**Base artifact:** latest Elsevier CAS single-column draft (with six clean replications incorporated).  
**Deliverable:** clean first-submission manuscript, not a claim about chronological study order.

## Verified build and citation checks

- All **40** bibliography entries are cited at least once; all in-text keys resolve; **zero** absent or unused keys.
- Replaced the hand-crafted 40-entry bibliography with Elsevier `cas-model2-names` BibTeX formatting, resolving repeated double punctuation and inconsistent author formatting.
- Verified core citation identities and available public publication metadata using linked publisher, conference, archive and indexing records; reference-specific source links and the scope of each check are in `REFERENCE_CROSSCHECK_40.csv`.
- The BibTeX build resolves all manuscript labels, figures and citations. References were checked at the level of publication identity and metadata, **not a complete sentence-by-sentence validation of every cited work's findings**.

## Bibliographic corrections implemented

1. Muré's ESAIM article is **2019**, not 2017 (the DOI includes an earlier year in its suffix). The key and citations were updated.
2. Chizat et al.'s paper is **Scaling algorithms for unbalanced optimal transport problems**; `optimal` was missing.
3. TabArena author names were corrected to Andrej Tschalzev and Prateek Mutalik Desai.
4. ScoringBench includes **Tizian Wenzel** as a third author.
5. Publisher DOI records were added for Hunter and Summerton (2006) and Ranjan and Gneiting (2010).
6. Brier's (1950) unusual DOI syntax is represented by an official publisher URL to avoid broken BibTeX/URL escaping.
7. TabICLv2, TabPFN, TabArena, ScoringBench and OpenML capitalization is preserved in the compiled bibliography.
8. PMLR metadata for TabICLv2, Majid et al., Guo et al. and Kull et al. were checked and enriched where useful.

## Writing and organization

- Removed manuscript revision-log framing and dated version labels in section titles; the paper reads as a consolidated first-submission manuscript.
- Preserved Protocol A (five-seed robustness), Protocol B (matched fixed-test study), and targeted replication as scientifically distinct, without portraying repeated folds as independent datasets.
- Rewrote the abstract, discussion, conclusion, contribution description and selected methodological transitions for clearer, less repetitive academic prose.
- Kept all audited numerical results and limitations, including 160 configurations, 80/80 structural changes, 67/80 increases in NLL, and the six exact clean replications.
- Preserved the truthful AI-assisted editorial disclosure. Natural-language editing is not evidence that any particular automated “AI detection” score will be achieved.

## Remaining submission requirements

- The S2 artifact does **not** yet have an immutable public DOI. Insert an actual DOI after publishing and licensing approval; do not invent it.
- MIC source ID 46980 remains unverified from the online query that returned HTTP 412; do not silently replace or relabel its data.
- Replication hashes fingerprint configured files for six new runs but do not retroactively prove all 160 earlier loaded model-weight bytes.
- Elsevier/CAS cover page has a class-level overfull-hbox diagnostic despite no text escaping the visible PDF page boundary in the renderer; inspect the title page in the journal proof system as an extra precaution.
- Confirm final CRediT, declarations, permissions and journal policy before uploading files.

## Notes on verification

External URLs are included for reference-level inspection. Records taken from DOI indexes are flagged in the CSV separately from publisher/proceedings records. The editorial audit does not constitute a plagiarism scan or a reliable classifier-based assessment of AI authorship.
