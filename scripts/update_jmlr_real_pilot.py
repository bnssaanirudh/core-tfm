from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


main_path = Path("paper/jmlr/main.tex")
text = main_path.read_text()

text = replace_once(
    text,
    "Controlled experiments with known conditional truth show that coherence, calibration, and probabilistic correctness are distinct. Exact-view perturbations identify when hard marginal preservation helps or hurts, while a heterogeneous task-mixture benchmark shows that Selective CoRe significantly outperforms the best fixed method and approaches a non-deployable candidate-family oracle. The results support selective rather than unconditional reconciliation and define a pre-specified real-TFM benchmark for final validation on released models.",
    "Controlled experiments with known conditional truth show that coherence, calibration, and probabilistic correctness are distinct. Exact-view perturbations identify when hard marginal preservation helps or hurts, while a heterogeneous task-mixture benchmark shows that Selective CoRe significantly outperforms the best fixed method and approaches a non-deployable candidate-family oracle. A five-fold TabICLv2 pilot on UCI Car Evaluation independently confirms substantial cross-query incompatibility and shows that the best unreconciled chain direction can still outperform both hard and soft reconciliation. The results support selective rather than unconditional reconciliation; a broader released-model matrix remains necessary before making average TFM performance claims.",
    "abstract",
)

text = replace_once(
    text,
    "\\item a heterogeneous 60-task benchmark in which Selective CoRe significantly outperforms the best fixed method and remains close to a full candidate-family oracle; and\n\\item a reproducible evaluation protocol covering joint, marginal, conditional, calibration, truth-distance, dependence-fidelity, and computational metrics.",
    "\\item a heterogeneous 60-task benchmark in which Selective CoRe significantly outperforms the best fixed method and remains close to a full candidate-family oracle;\n\\item a five-fold released-model pilot on TabICLv2 and UCI Car Evaluation that reproduces the qualitative magnitude of inconsistency while showing that unconditional reconciliation need not improve joint log loss; and\n\\item a reproducible evaluation protocol covering joint, marginal, conditional, calibration, truth-distance, dependence-fidelity, and computational metrics.",
    "contributions",
)

old_section = r'''\section{Real-TFM Benchmark Protocol and Current Limitation}
The controlled evidence establishes how reconciliation behaves when probability-view reliability can be measured or manipulated. It does not yet establish an average improvement on released foundation models. A top-tier submission under the current title therefore requires a completed real-TFM matrix before submission.

The pre-specified model set is TabPFN-3 \citep{grinsztajn2026tabpfn3}, TabICLv2 \citep{qu2026tabiclv2}, and TabFM \citep{google2026tabfm}. The initial data set list follows the categorical target pairs of \citet{klotergens2026agree}: Anneal, Credit, Phishing, MIC, Customer, Car, Marketing, Wine, Nursery, and Diamonds. Each outer fold must extract all four probability views; an inner split chooses direction weight, marginal penalty, or Selective CoRe policy; no test information may influence selection. Cross-dataset comparisons use data set--model cells as statistical units rather than treating individual rows as independent replications.

As an infrastructure sanity route, the UCI Car Evaluation data provide 1,728 rows, six categorical features, class as target, and safety as a categorical feature \citep{bohanec1988car}. This pair matches one of the planned real-data evaluations and is licensed for redistribution with attribution. Real-model numbers will be inserted only after checkpoint inference completes and package/checkpoint versions are frozen. Until that matrix is complete, this document is a formatted working draft rather than a submission-ready manuscript.
'''
new_section = r'''\section{Real-TFM Pilot and Benchmark Protocol}
The controlled evidence establishes how reconciliation behaves when probability-view reliability can be measured or manipulated. We now add a first released-model pilot, while keeping the broader model--data set matrix pre-specified rather than claiming that one cell establishes average TFM gains.

\subsection{Five-fold TabICLv2 pilot on Car Evaluation}
We evaluate TabICLv2 on UCI Car Evaluation, using $A=$ class and $B=$ safety, with the remaining five categorical attributes as $X$ \citep{bohanec1988car}. The data contain 1,728 rows. We use five-fold stratified cross-validation on $A$ with shuffle enabled and random seed 42. For CPU tractability, the pilot uses one TabICLv2 estimator, key--value caching, and the released 2026 checkpoint. This is therefore not an exact reproduction of \citet{klotergens2026agree}: their fold seed is not published and their default inference configuration differs.

Across the five folds, mean factorization inconsistency is $0.06931\pm0.00354$ TV. The corresponding value reported by \citet{klotergens2026agree} for TabICLv2 on Car is $0.0644\pm0.0058$. Given the non-identical split and inference configuration, we treat this only as a qualitative reproduction of the inconsistency scale, not as a numerical replication.

Table~\ref{tab:realpilot} reports joint NLL and marginal distortion. The key result is negative but informative: the $B\!\rightarrow\!A$ factorization is the best mean-NLL method on every aggregate comparison, at $1.12089\pm0.00471$. Geometric pooling reaches $1.13434\pm0.00786$, Soft CoRe with $\lambda_A=\lambda_B=1$ reaches $1.13840\pm0.00982$, and Hard CoRe reaches $1.20412\pm0.02819$. Thus the released model exhibits clear incompatibility, yet enforcing a single repaired joint does not improve the best raw factorization. Hard CoRe nearly eliminates marginal distortion by construction, but that structural success coincides with worse predictive score. This is exactly the distinction between compatibility and correctness that motivates Selective CoRe.

\begin{table}[t]
\centering
\caption{Five-fold TabICLv2 pilot on UCI Car Evaluation. Values are fold mean $\pm$ sample standard deviation. Lower is better for both metrics. This CPU pilot uses one estimator and is not an exact reproduction of prior published settings.}
\label{tab:realpilot}
\begin{tabular}{lcc}
\toprule
Method & Joint NLL & Marginal distortion \\
\midrule
$B\!\rightarrow\!A$ ($\Jone$) & $1.12089\pm0.00471$ & $0.02495\pm0.00294$ \\
$A\!\rightarrow\!B$ ($\Jtwo$) & $1.18400\pm0.00922$ & $0.02162\pm0.00118$ \\
Independent & $1.55117\pm0.01643$ & $0.00000\pm0.00000$ \\
Arithmetic pool & $1.14625\pm0.00559$ & $0.02329\pm0.00205$ \\
Geometric pool & $1.13434\pm0.00786$ & $0.02762\pm0.00177$ \\
Hard CoRe & $1.20412\pm0.02819$ & $<2\times10^{-13}$ \\
Soft CoRe ($\lambda=1$) & $1.13840\pm0.00982$ & $0.02253\pm0.00183$ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Pre-specified broader benchmark}
The pre-specified model set remains TabPFN-3 \citep{grinsztajn2026tabpfn3}, TabICLv2 \citep{qu2026tabiclv2}, and TabFM \citep{google2026tabfm}. The initial data set list follows the categorical target pairs of \citet{klotergens2026agree}: Anneal, Credit, Phishing, MIC, Customer, Car, Marketing, Wine, Nursery, and Diamonds. Each outer fold must extract all four probability views; an inner split chooses direction weight, marginal penalty, or Selective CoRe policy; no test information may influence selection. Cross-dataset comparisons use data set--model cells as statistical units rather than treating individual rows as independent replications.

The pilot is sufficient to demonstrate that the implemented four-view extraction and reconciliation pipeline operates on a released TFM, but it is not sufficient for an average-performance claim. Until the broader matrix is complete, this document remains a working draft rather than a submission-ready manuscript.
'''
text = replace_once(text, old_section, new_section, "real-TFM section")

text = replace_once(
    text,
    "Third, exact-view and surrogate experiments provide causal diagnostic control but cannot replace released-model validation.",
    "Third, the five-fold TabICLv2 Car pilot establishes released-model feasibility but covers only one model--data set cell; exact-view and surrogate experiments therefore still cannot replace the planned broader released-model validation.",
    "limitations",
)

text = replace_once(
    text,
    "The real-data release will archive raw data snapshots by source identifier and hash, model package versions, checkpoint identifiers, license/access conditions, hardware, inference configuration, raw fold-level outputs, and scripts that regenerate every table and figure.",
    "The repository now includes fold-level output for the five-fold TabICLv2 Car pilot. The complete real-data release will additionally archive raw data snapshots by source identifier and hash, model package versions, checkpoint identifiers, license/access conditions, hardware, inference configuration, and scripts that regenerate every table and figure.",
    "reproducibility",
)

text = replace_once(
    text,
    "In heterogeneous tasks, Selective CoRe significantly improves over the best fixed policy and approaches the oracle candidate family as validation evidence grows. The remaining decisive step is the pre-specified benchmark on released TFMs. Only after that benchmark is complete should the paper claim average gains for deployed foundation models.",
    "In heterogeneous tasks, Selective CoRe significantly improves over the best fixed policy and approaches the oracle candidate family as validation evidence grows. A five-fold TabICLv2 Car pilot adds an important real-model counterexample: incompatibility is substantial, but the best raw chain direction still beats pooling and reconciliation in joint log loss. The remaining decisive step is the full pre-specified benchmark across released TFMs and data sets. Only after that matrix is complete should the paper claim average gains for deployed foundation models.",
    "conclusion",
)

main_path.write_text(text)

readme_path = Path("README.md")
readme = readme_path.read_text()
readme = replace_once(
    readme,
    "These are controlled method results, **not yet released-TFM performance claims**.",
    "The repository also contains a **five-fold released-model pilot** on TabICLv2 + UCI Car Evaluation. Mean factorization TV is `0.06931 ± 0.00354`; the best mean joint NLL is the unreconciled `B→A` chain (`1.12089`), ahead of geometric pooling (`1.13434`), Soft CoRe with `lambda=1` (`1.13840`), and Hard CoRe (`1.20412`). This supports selective rather than unconditional repair. It is one model–dataset cell, not an average released-TFM performance claim.",
    "README real evidence",
)
readme_path.write_text(readme)

print("Updated JMLR manuscript and README with five-fold TabICLv2 Car pilot.")
