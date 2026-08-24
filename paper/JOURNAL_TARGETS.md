# Journal targeting strategy

## Recommended current target: Transactions on Machine Learning Research (TMLR)

TMLR remains the best fit. Its official venue description explicitly emphasizes
technical correctness over subjective significance and supports rolling,
double-blind review. CoRe-TFM now has a complete negative/conditional real-model
result rather than an incomplete matrix: incompatibility is measurable, marginal
distortion can be reduced, but the 48-candidate selector loses to arithmetic pooling
under a 52-label validation budget and reverses sign by upstream model. That
technically careful behavior study fits TMLR better than a performance-led claim.

Official scope: https://openreview.net/group?id=TMLR

## Retained alternative: Journal of Machine Learning Research (JMLR)

JMLR remains aspirational. Its official author guidance requires clearly supported
claims, reproducible experiments, and broad machine-learning interest. Before a
JMLR submission, complete independent sampling-seed and training-context
sensitivity, resolve the rare-class dataset design, and preferably add a third
released TFM. The current negative finding is acceptable in principle, but the
evidence must demonstrate that the model interaction is stable beyond one bounded
sampling realization.

Official author information: https://www.jmlr.org/author-info.html

## Secondary target: Machine Learning (Springer Nature)

A Springer `sn-jnl` adaptation is retained as a strong secondary target. *Machine Learning* explicitly accepts methodology papers but requires precise, replicable support for claims. The scientific content must remain synchronized with the evidence package rather than diverging into a separate paper.

## Non-negotiable claim rules

1. Do not use post-repair zero inconsistency as the headline empirical result.
2. Do not claim invention of conditional compatibility, KL projection, IPF, or Sinkhorn scaling.
3. Do not claim first post-hoc TFM probability correction; DistPFN already addresses label shift.
4. Cite broader conditional-consistency theory, including Majid et al. (AISTATS 2025).
5. State the completed bounded benchmark accurately; do not claim an average
   Selective-CoRe gain, because the primary result significantly favors arithmetic.
6. Keep proper scoring, conditional performance, calibration, distortion, and truth-distance separate.
7. Never label the 30 dataset--model cells as independent confirmatory units; the
   primary analysis uses ten datasets after averaging the two TFMs.
