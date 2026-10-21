# Journal-fit evaluation: CoRe-TFM (20 October 2026)

## Manuscript considered
Coherence Is Not Accuracy: Evaluating Reconciliation, Calibration, and Reliability-Aware View Selection in Tabular Foundation Models.

The paper is a methodological and empirical reliability study, not a proven new accuracy-improving reconciliation algorithm. Its strongest supported result is the disconnect between marginal consistency and proper predictive scoring across two TFMs and ten datasets. The archival audit verifies 160 configurations; six subsequent GPU re-executions exactly reproduced saved probabilities. The guarded repair policy never activated. The exploratory Holm-adjusted tests do not confirm that forced repair harms all population settings.

## Ranked journals

1. **Knowledge-Based Systems (Elsevier)** — recommended first submission. Journal scope expressly includes machine-learning theory, methodologies, prediction systems, and intelligent decision support. This is the closest match to the paper's evaluation of reliability-sensitive post-processing of model outputs. Publisher page reports 7.6 Journal Impact Factor and 13.7 CiteScore. **Fit assessment: 8.8/10** (editorial judgment, not acceptance probability). Main risk: convincing an editor of original scientific contribution despite lack of positive CoRe accuracy gains. Emphasize falsifiable reliability assumptions, strict-propriety counterexample, multi-protocol benchmark, and actionable diagnostics. Source: https://shop.elsevier.com/journals/knowledge-based-systems/0950-7051 ; metric: https://www.sciencedirect.com/journal/knowledge-based-systems/about/policies

2. **Information Sciences (Elsevier)** — strongest broad information-theory/uncertainty alternative. Its scope covers innovative information science and intelligent systems, including formal reasoning about probability representations. **Fit: 8.0/10.** Main risk: editorial expectation of a more general mathematical or algorithmic advance; currently the paper is strongest as a detailed empirical investigation. Source: https://shop.elsevier.com/journals/information-sciences/0020-0255

3. **Neurocomputing (Elsevier)** — good machine-learning reliability venue. It explicitly welcomes learning-system theory, software practice, and applications. **Fit: 7.6/10.** Main risk: less directly concerned with probabilistic information compatibility; highlight reproducible model-centric diagnosis. Source: https://shop.elsevier.com/journals/neurocomputing/0925-2312

4. **Expert Systems with Applications (Elsevier)** — plausible if framed as practical decision-support safety or prediction reliability. **Fit: 6.9/10.** Main risk: the current paper lacks a prospective domain-specific deployment or downstream utility experiment. Source: https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174

5. **Information Fusion (Elsevier)** — aspirational/high-risk option. Its scope includes decision fusion and multi-classifier integration, but the paper's primary result is a critique of forced probability fusion, not a decisively superior fusion scheme. **Fit: 6.4/10.** A publisher listing reported an impact factor of 17.4. This should not drive a first submission unless theory and fusion novelty are materially strengthened. Source: https://shop.elsevier.com/journals/information-fusion/1566-2535 ; metric context: https://www.sciencedirect.com/browse/calls-for-papers?subject=computer-science

Scores above are subjective editorial fit assessments, not quartile certifications or acceptance-rate estimates. Journal metrics and individual author requirements can change; verify the live guide for authors when submitting.

## Pre-submission actions (journal independent)

- Publish the approved, license-reviewed prediction/audit evidence (S2) and, if possible, the distinct historical evidence (S1) in an immutable repository. Add the actual DOI(s), not a placeholder DOI, to the Data Availability statement.
- Include the standalone manuscript PDF with all readable scientific appendices. Keep machine-readable caches and manifests as a separate repository dataset, not another supplementary PDF unless the journal specifically asks.
- Inspect the bibliography against 40 original publisher/DOI records; earlier drafts included reconstructed metadata and not every entry has been reverified individually.
- Resolve the MIC OpenML 46980 raw metadata failure or retain a transparent 10/11 source-verification limitation.
- Verify author affiliations, full CRediT roles, funding, conflicts, and publisher-specific AI assistance disclosure in the submission system. The author approval reported in the conversation is accepted but is not a substitute for individual online confirmations if requested.
- For Knowledge-Based Systems, check current manuscript length/figure preferences, highlights, graphical abstract requirements, and any data policy or reporting checklist before upload.

## Recommended submission positioning

Avoid promising an accuracy improvement from CoRe. Present the paper as a carefully controlled study of a ubiquitous failure mode in post-hoc prediction repair. Contrast structurally successful projection with worsening proper scores, test whether validation-guided selection is competitive, and show why no-inference from disagreement magnitude alone is justified. State the six exact clean replications and the remaining historical checkpoint/source uncertainty precisely.
