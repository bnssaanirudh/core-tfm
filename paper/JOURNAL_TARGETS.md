# Journal targeting strategy

## Primary scientific target: Journal of Machine Learning Research (JMLR)

JMLR is the strictest current target used to shape the master manuscript because it requires its official `jmlr2e` style at submission and evaluates principled algorithms with sound empirical validation. The repository therefore keeps the current master draft in `paper/jmlr/`.

The JMLR draft is intentionally **not submission-ready** until the released-TFM benchmark is complete and author affiliation/disclosures are final. Controlled evidence alone should not be described as a demonstrated average improvement on TabPFN-3, TabICLv2, or TabFM.

## Secondary target: Machine Learning (Springer Nature)

A Springer `sn-jnl` adaptation is retained as the realistic Q1/Scopus secondary target. The scientific content should be generated from the same evidence and claims as the JMLR master rather than diverging into a separate paper.

## Non-negotiable claim rules

1. Do not use post-repair zero inconsistency as the headline empirical result.
2. Do not claim invention of conditional compatibility, KL projection, IPF, or Sinkhorn scaling.
3. Do not claim first post-hoc TFM probability correction; DistPFN already addresses label shift.
4. Cite broader conditional-consistency theory, including Majid et al. (AISTATS 2025).
5. Do not claim real-TFM gains until the real benchmark is complete.
6. Keep proper scoring, conditional performance, calibration, distortion, and truth-distance separate.
