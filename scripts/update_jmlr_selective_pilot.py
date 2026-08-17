from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


main_path = Path("paper/jmlr/main.tex")
text = main_path.read_text()

text = replace_once(
    text,
    "A five-fold TabICLv2 pilot on UCI Car Evaluation independently confirms substantial cross-query incompatibility and shows that the best unreconciled chain direction can still outperform both hard and soft reconciliation. The results support selective rather than unconditional reconciliation; a broader released-model matrix remains necessary before making average TFM performance claims.",
    "A five-fold TabICLv2 pilot on UCI Car Evaluation independently confirms substantial cross-query incompatibility and shows that the best unreconciled chain direction can still outperform both hard and soft reconciliation. A leakage-free inner-validation Selective CoRe policy chooses that raw direction in four of five outer folds and incurs only 0.00027 mean joint-NLL regret to the per-fold best original direction. The results support selective rather than unconditional reconciliation; a broader released-model matrix remains necessary before making average TFM performance claims.",
    "abstract",
)

text = replace_once(
    text,
    "Table~\\ref{tab:realpilot} reports joint NLL and marginal distortion. The key result is negative but informative: the $B\\!\\rightarrow\\!A$ factorization is the best mean-NLL method on every aggregate comparison, at $1.12089\\pm0.00471$. Geometric pooling reaches $1.13434\\pm0.00786$, Soft CoRe with $\\lambda_A=\\lambda_B=1$ reaches $1.13840\\pm0.00982$, and Hard CoRe reaches $1.20412\\pm0.02819$. Thus the released model exhibits clear incompatibility, yet enforcing a single repaired joint does not improve the best raw factorization. Hard CoRe nearly eliminates marginal distortion by construction, but that structural success coincides with worse predictive score. This is exactly the distinction between compatibility and correctness that motivates Selective CoRe.",
    "Table~\\ref{tab:realpilot} reports joint NLL and marginal distortion. The key fixed-policy result is negative but informative: the $B\\!\\rightarrow\\!A$ factorization is the best mean-NLL method on every aggregate comparison, at $1.12089\\pm0.00471$. Geometric pooling reaches $1.13434\\pm0.00786$, Soft CoRe with $\\lambda_A=\\lambda_B=1$ reaches $1.13840\\pm0.00982$, and Hard CoRe reaches $1.20412\\pm0.02819$. Thus the released model exhibits clear incompatibility, yet enforcing a single repaired joint does not improve the best raw factorization. Hard CoRe nearly eliminates marginal distortion by construction, but that structural success coincides with worse predictive score.\n\nWe therefore run the full Selective CoRe candidate family using a leakage-free inner split: 20\\% of each outer-training fold is held out for policy selection, the policy is frozen, all four probability views are refit on the complete outer-training fold, and the untouched outer test fold is scored. Selective CoRe chooses the raw $B\\!\\rightarrow\\!A$ factorization in four folds and Soft CoRe with $w=1,\\lambda=0.1$ in one fold. Its mean test joint NLL is $1.12116\\pm0.00485$, only $0.00027$ above the per-fold best original direction. This is the intended behavior of selective reconciliation: validation mostly rejects unnecessary repair and pays only a small cost in the one fold where it selects a mild repair.",
    "real pilot paragraph",
)

text = replace_once(
    text,
    "Soft CoRe ($\\lambda=1$) & $1.13840\\pm0.00982$ & $0.02253\\pm0.00183$ \\\\",
    "Soft CoRe ($\\lambda=1$) & $1.13840\\pm0.00982$ & $0.02253\\pm0.00183$ \\\\\nSelective CoRe & $1.12116\\pm0.00485$ & $0.02495\\pm0.00294$ \\\\",
    "pilot table",
)

text = replace_once(
    text,
    "A five-fold TabICLv2 Car pilot adds an important real-model counterexample: incompatibility is substantial, but the best raw chain direction still beats pooling and reconciliation in joint log loss. The remaining decisive step is the full pre-specified benchmark across released TFMs and data sets.",
    "A five-fold TabICLv2 Car pilot adds an important real-model counterexample: incompatibility is substantial, but the best raw chain direction still beats unconditional pooling and reconciliation in joint log loss. Leakage-free Selective CoRe responds appropriately by choosing the raw direction in four of five folds and nearly matching the per-fold best original score overall. The remaining decisive step is the full pre-specified benchmark across released TFMs and data sets.",
    "conclusion",
)

main_path.write_text(text)

readme_path = Path("README.md")
readme = readme_path.read_text()
readme = replace_once(
    readme,
    "This supports selective rather than unconditional repair. It is one model–dataset cell, not an average released-TFM performance claim.",
    "A leakage-free five-fold Selective CoRe run chooses the raw `B→A` direction in 4/5 folds and a mild Soft CoRe policy once, reaching `1.12116 ± 0.00485` mean joint NLL—only `0.00027` above the per-fold best original direction. This supports selective rather than unconditional repair. It is one model–dataset cell, not an average released-TFM performance claim.",
    "README",
)
readme_path.write_text(readme)

print("Updated manuscript and README with leakage-free Selective CoRe real-model result.")
