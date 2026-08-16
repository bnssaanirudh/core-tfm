from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib import colors

OUT = "paper/CoRe_TFM_Public_Working_Draft.pdf"
W, H = A4
M = 48

c = canvas.Canvas(OUT, pagesize=A4, pageCompression=0)
c.setTitle("CoRe-TFM: Four-View Post-Hoc Probability Reconciliation for Tabular Foundation Models")
c.setAuthor("Badampudi Agasthya Anirudh")
page = 1
y = H - M


def footer():
    c.setFont("Times-Roman", 8)
    c.setFillColor(colors.grey)
    c.drawCentredString(W / 2, 24, f"{page}")
    c.setFillColor(colors.black)


def newpage():
    global page, y
    footer()
    c.showPage()
    page += 1
    y = H - M


def need(h):
    if y - h < 42:
        newpage()


def clean(s):
    repl = {
        "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u00a0": " ", "\u03b3": "gamma",
        "\u03c1": "rho", "\u03bb": "lambda", "\u2248": "~", "\u2260": "!=",
        "\u2264": "<=", "\u2265": ">=",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s.encode("ascii", "replace").decode("ascii")


def wrap(text, font="Times-Roman", size=9.2, width=W - 2 * M):
    words = clean(text).split()
    lines, cur = [], ""
    for word in words:
        candidate = word if not cur else cur + " " + word
        if stringWidth(candidate, font, size) <= width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def para(text, size=9.2, leading=11.2, after=5, font="Times-Roman"):
    global y
    lines = wrap(text, font, size)
    need(len(lines) * leading + after)
    c.setFont(font, size)
    for line in lines:
        c.drawString(M, y, line)
        y -= leading
    y -= after


def heading(text, level=1):
    global y
    size = {1: 13, 2: 11, 3: 9.8}.get(level, 9.8)
    need(size + 13)
    y -= 3
    c.setFont("Helvetica-Bold", size)
    c.drawString(M, y, clean(text))
    y -= size + 7


def bullet(text):
    global y
    lines = wrap(text, "Times-Roman", 9.0, W - 2 * M - 14)
    need(len(lines) * 10.8 + 3)
    c.setFont("Times-Roman", 9)
    c.drawString(M + 2, y, "-")
    for line in lines:
        c.drawString(M + 14, y, line)
        y -= 10.8
    y -= 2


def equation(text):
    global y
    need(22)
    c.setFont("Courier", 8.5)
    c.drawCentredString(W / 2, y, clean(text))
    y -= 20


def draft_box():
    global y
    text = (
        "WORKING DRAFT - NOT READY FOR SUBMISSION. Controlled synthetic evidence is included; "
        "real TabPFN-3, TabICLv2, and TabFM experiments remain mandatory."
    )
    lines = wrap(text, "Helvetica-Bold", 8.2, W - 2 * M - 12)
    h = 12 + len(lines) * 10
    c.setFillColorRGB(1, 0.96, 0.80)
    c.rect(M, y - h + 4, W - 2 * M, h, fill=1, stroke=1)
    c.setFillColor(colors.black)
    yy = y - 8
    c.setFont("Helvetica-Bold", 8.2)
    for line in lines:
        c.drawString(M + 6, yy, line)
        yy -= 10
    y -= h + 8


def table(headers, rows, widths=None, font_size=7.2):
    global y
    n = len(headers)
    if widths is None:
        widths = [(W - 2 * M) / n] * n
    rh = 15
    need((len(rows) + 1) * rh + 10)
    x, top = M, y
    c.setFont("Helvetica-Bold", font_size)
    for j, header in enumerate(headers):
        c.setFillColorRGB(0.93, 0.93, 0.93)
        c.rect(x, top - rh, widths[j], rh, fill=1, stroke=1)
        c.setFillColor(colors.black)
        c.drawString(x + 3, top - 10, clean(str(header))[:40])
        x += widths[j]
    yy = top - rh
    c.setFont("Times-Roman", font_size)
    for row in rows:
        x = M
        for j, value in enumerate(row):
            c.rect(x, yy - rh, widths[j], rh, fill=0, stroke=1)
            c.drawString(x + 3, yy - 10, clean(str(value))[:45])
            x += widths[j]
        yy -= rh
    y = yy - 8


def linechart(title, xvals, series, ylabel):
    global y
    h = 155
    need(h + 25)
    x0, y0 = M + 42, y - h + 30
    ww, hh = W - 2 * M - 65, h - 55
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W / 2, y, clean(title))
    y -= 12
    c.setStrokeColor(colors.black)
    c.line(x0, y0, x0, y0 + hh)
    c.line(x0, y0, x0 + ww, y0)
    all_values = [v for vals in series.values() for v in vals]
    ymin, ymax = min(all_values), max(all_values)
    pad = (ymax - ymin) * 0.12 or 0.1
    ymin, ymax = ymin - pad, ymax + pad
    cols = [colors.black, colors.darkgrey, colors.grey, colors.Color(0.25, 0.25, 0.25)]
    for k, (name, vals) in enumerate(series.items()):
        pts = []
        for i, value in enumerate(vals):
            xx = x0 + (i / (len(xvals) - 1)) * ww if len(xvals) > 1 else x0 + ww / 2
            yy = y0 + (value - ymin) / (ymax - ymin) * hh
            pts.append((xx, yy))
        c.setStrokeColor(cols[k % len(cols)])
        c.setLineWidth(1.3 if k == 0 else 0.8)
        for a, b in zip(pts, pts[1:]):
            c.line(a[0], a[1], b[0], b[1])
        for xx, yy in pts:
            c.circle(xx, yy, 2, fill=1, stroke=1)
    c.setFillColor(colors.black)
    c.setFont("Times-Roman", 7)
    for i, xv in enumerate(xvals):
        xx = x0 + (i / (len(xvals) - 1)) * ww if len(xvals) > 1 else x0 + ww / 2
        c.drawCentredString(xx, y0 - 11, str(xv))
    c.saveState()
    c.translate(M + 8, y0 + hh / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, clean(ylabel))
    c.restoreState()
    lx, ly = x0 + 6, y0 + hh - 8
    for k, name in enumerate(series):
        c.setStrokeColor(cols[k % len(cols)])
        c.line(lx, ly, lx + 14, ly)
        c.setFillColor(colors.black)
        c.drawString(lx + 18, ly - 2, clean(name))
        ly -= 9
    y = y0 - 24


c.setFont("Helvetica-Bold", 15)
title = "CoRe-TFM: Four-View Post-Hoc Probability Reconciliation for Tabular Foundation Models"
for line in wrap(title, "Helvetica-Bold", 15, W - 2 * M):
    c.drawCentredString(W / 2, y, line)
    y -= 18
y -= 5
c.setStrokeColorRGB(0.15, 0.35, 0.65)
c.line(M, y, W - M, y)
c.setStrokeColor(colors.black)
y -= 22
c.setFont("Helvetica-Bold", 10)
c.drawCentredString(W / 2, y, "Badampudi Agasthya Anirudh")
y -= 14
c.setFont("Times-Roman", 9)
c.drawCentredString(W / 2, y, "Integrated M.Tech (Data Science) - affiliation to be completed")
y -= 19
draft_box()

heading("Abstract")
para("Tabular foundation models (TFMs) are usually trained as univariate predictors but are increasingly composed autoregressively for multivariate inference. Recent work shows that contemporary TFMs can violate marginalization and factorization consistency: direct marginals and conditional predictions obtained from the same frozen model need not correspond to one joint distribution. We study post-hoc reconciliation of these incompatible predictions without retraining. CoRe-TFM treats p(A|X), p(B|X), p(A|B,X), and p(B|A,X) as four probabilistic views of a categorical target pair. Hard CoRe projects a consensus joint onto the transportation polytope defined by the direct marginals. Soft CoRe relaxes marginal preservation and admits generalized Sinkhorn-style multiplicative updates. Selective CoRe uses validation data to decide whether reconciliation should be applied. Controlled experiments with exact ground-truth joints across 10 seeds show that coherence is not equivalent to probabilistic correctness: the preferred method changes with dependence, sample size, feature dimension, class cardinality, and imbalance. A final real-model benchmark on TabPFN-3, TabICLv2, and TabFM remains required before submission.")
para("Keywords: tabular foundation models; probabilistic consistency; probability reconciliation; calibration; uncertainty quantification; Sinkhorn scaling", font="Times-Bold", size=8.8)

heading("1 Introduction")
para("Tabular foundation models such as TabPFN, TabICL, and TabFM perform in-context prediction from labeled context sets and expose probability predictions without fitting a conventional task-specific model. Their probabilistic outputs make them attractive for uncertainty quantification, conditional density estimation, imputation, sequential sampling, and other multivariate uses.")
para("A difficulty appears when a model designed for univariate prediction is used as a multivariate building block. For two categorical targets A and B, the same frozen TFM can provide direct marginals p(A|X) and p(B|X), plus conditionals p(A|B,X) and p(B|A,X). If these four views came from one coherent joint distribution, the two chain-rule factorizations would agree and marginalization would recover the direct predictions. Recent work documents systematic violations of these requirements in contemporary TFMs and explicitly identifies post-hoc reconciliation as an open direction.")
para("Coherence alone is not enough: any single normalized joint distribution is internally coherent by construction. The research problem is therefore to select a coherent joint that does not destroy useful predictive information. We ask whether reconciliation improves proper scoring, calibration, conditional predictions, and distance to known truth, and whether validation data can identify situations in which repair should be skipped.")
heading("1.1 Research questions and contributions", 2)
for item in [
    "RQ1 - Reconciliation: how can four incompatible TFM prediction views be combined into a single joint without retraining the foundation model?",
    "RQ2 - Correctness: when the true conditional joint is known, does reconciliation move predictions closer to truth or can it worsen an already good ordering?",
    "RQ3 - Selectivity: can validation evidence identify a consistency dividend and avoid regimes that impose a consistency tax?",
    "Contribution 1: a TFM-specific four-view convex reconciliation formulation with hard and soft marginal constraints.",
    "Contribution 2: generalized Sinkhorn-style updates that make the soft objective fast enough for batched post-processing.",
    "Contribution 3: a selective validation protocol plus known-truth synthetic benchmarks measuring coherence, fidelity, calibration, and proper scores.",
    "Contribution 4: negative evidence showing that a simple label-free directional-defect heuristic is not a reliable universal repair trigger.",
]:
    bullet(item)

heading("2 Related work and novelty boundary")
para("The work sits at the intersection of tabular foundation models, probabilistic reliability, incompatible conditional distributions, and KL projection. Recent TFM studies increasingly evaluate calibration and uncertainty rather than accuracy alone. Compatibility of separately specified conditional distributions is much older than foundation models, and iterative proportional fitting, Bregman projection, and generalized or unbalanced Sinkhorn scaling are established optimization tools. Post-hoc adjustment of TFM probabilities also exists for other failure modes such as label shift.")
para("Accordingly, CoRe-TFM does not claim to invent probability compatibility, KL projection, Sinkhorn scaling, or generic post-hoc TFM correction. Its intended contribution is the four-view TFM reconciliation formulation, the selective deployment protocol, and an empirical characterization of the coherence-fidelity-calibration trade-off raised by the newly documented TFM inconsistency problem.")

heading("3 Problem formulation")
para("For context dataset D, test covariates x, and categorical targets A and B, query the frozen predictor four ways:")
equation("p_A(a)=p(A=a|x,D),   p_B(b)=p(B=b|x,D)")
equation("p_A|B(a|b)=p(A=a|B=b,x,D),   p_B|A(b|a)=p(B=b|A=a,x,D)")
para("The two autoregressive joints are")
equation("J1(a,b)=p_B(b) p_A|B(a|b),    J2(a,b)=p_A(a) p_B|A(b|a).")
para("Factorization inconsistency is measured by total variation TV(J1,J2). Each direction also has a non-trivial marginalization defect because one marginal is guaranteed by construction while the other need not agree with its direct TFM prediction.")

heading("4 CoRe-TFM")
heading("4.1 KL barycenter baselines", 2)
para("Arithmetic and geometric pooling are retained as strong baselines rather than arbitrary averaging rules. A weighted arithmetic pool minimizes a forward-KL barycenter objective, whereas the normalized weighted geometric pool minimizes the reverse-KL barycenter objective.")
equation("Q_arith = w J1 + (1-w) J2")
equation("M_w(a,b) proportional to J1(a,b)^w J2(a,b)^(1-w)")
heading("4.2 Hard CoRe", 2)
para("Hard CoRe trusts the direct marginals and reconciles only the dependence structure by projecting the geometric consensus onto the transportation polytope with row sums p_A and column sums p_B:")
equation("Q_H = argmin_Q KL(Q || M_w)  subject to Q_A=p_A and Q_B=p_B.")
para("The feasible set is non-empty because the independent joint p_A p_B^T always satisfies the marginal constraints. Under positive support, strict convexity yields a unique optimum. Iterative proportional fitting provides the numerical solution, and direct marginal decisions remain exactly unchanged.")
heading("4.3 Soft four-view CoRe", 2)
para("Exact marginal preservation can preserve a marginal error. Soft CoRe therefore optimizes")
equation("min_Q w KL(Q||J1)+(1-w) KL(Q||J2)+lambda_A KL(Q_A||p_A)+lambda_B KL(Q_B||p_B).")
para("The KL chain rule decomposes each joint term into a marginal term plus an expected conditional term. Consequently the objective simultaneously reconciles p(A|X), p(B|X), p(A|B,X), and p(B|A,X), rather than merely averaging two joint matrices.")
heading("4.4 Generalized Sinkhorn updates", 2)
para("Writing M=M_w, the optimum has a matrix-scaling form Q proportional to diag(u) M diag(v). With tau_A=lambda_A/(1+lambda_A) and tau_B=lambda_B/(1+lambda_B), alternating multiplicative updates are:")
equation("u <- (p_A / (M v))^tau_A,      v <- (p_B / (M^T u))^tau_B.")
para("An independent exponentiated-gradient solver and generic constrained optimizer are retained as numerical cross-checks in small problems.")
heading("4.5 Selective CoRe", 2)
para("Repair is treated as a decision, not a mandatory transformation. An inner validation split selects among an original factorization, arithmetic or geometric pooling, hard reconciliation, and soft reconciliation. The selected rule is then evaluated only on an untouched outer test fold. This prevents a method from receiving credit for consistency when it causes a predictive consistency tax.")

heading("5 Experimental design")
para("The current evidence is a checkpoint-free controlled benchmark designed to validate the research methodology before expensive TFM runs. Synthetic categorical data-generating processes provide exact P*(A,B|X), allowing direct measurement of whether reconciliation approaches the true joint. One factor is varied at a time across 10 random seeds: dependence strength gamma, sample size, feature dimension, class cardinality, and class imbalance.")
table(["Axis", "Values"], [
    ("Dependence gamma", "0, 0.5, 1.5, 3.0"),
    ("Sample size n", "250, 1000, 5000"),
    ("Feature dimension d", "5, 20, 50"),
    ("Classes per target", "2, 3, 5"),
    ("Majority probability", "0.50, 0.75, 0.90"),
    ("Seeds", "10 per condition"),
], [155, W - 2 * M - 155], 7.5)
para("Metrics include original factorization TV; joint NLL and Brier score; top-label ECE; induced conditional NLL/Brier/ECE in both directions; TV and Jensen-Shannon distance to known truth; mutual-information error; marginal and reconciliation distortion; and runtime. Real-data evaluation will use deterministic five-fold testing with inner validation for adaptive choices.")

heading("6 Controlled empirical results")
para("The key result is regime dependence rather than universal superiority. As dependence strengthens, the independent joint becomes increasingly inappropriate and more structured reconciliation can become useful. At strong dependence, adaptive Soft CoRe improves mean joint NLL relative to arithmetic pooling and hard marginal-preserving reconciliation in the current 10-seed study. In small-sample conditions, simple arithmetic pooling can remain preferable.")
linechart("Joint NLL changes with dependence", [0, 0.5, 1.5, 3.0], {
    "Adaptive Soft": [1.836, 1.823, 1.743, 1.5854],
    "Arithmetic": [1.837, 1.8257, 1.751, 1.6082],
    "Hard CoRe": [1.838, 1.826, 1.753, 1.6048],
    "Independent": [1.8365, 1.821, 1.801, 1.7435],
}, "Joint negative log likelihood")
para("At gamma=3, the detailed means are:")
table(["Method", "Joint NLL", "Conditional NLL", "TV to truth", "MI abs. error"], [
    ("Validation-selected order", "1.5865", "0.7148", "0.2339", "0.1276"),
    ("Arithmetic", "1.6082", "0.7380", "0.2492", "0.1859"),
    ("Geometric", "1.6034", "0.7300", "0.2448", "0.1758"),
    ("Hard CoRe", "1.6048", "0.7331", "0.2456", "0.1738"),
    ("Adaptive Soft CoRe", "1.5854", "0.7135", "0.2346", "0.1426"),
    ("Selective CoRe", "1.5861", "0.7136", "0.2351", "0.1430"),
    ("Independent", "1.7435", "0.8718", "0.3106", "0.2466"),
], [125, 72, 85, 72, 78], 6.6)
para("The controlled experiments also show that raw inconsistency magnitude is not a universal repair-needed score. In the dependence sweep, factorization TV has essentially no rank correlation with Adaptive Soft CoRe gain over a validation-selected original order (rho about 0.065). It has a stronger positive association with the gain over naive arithmetic pooling (rho about 0.422, p about 0.0067). Thus larger inconsistency can signal that symmetric averaging is inadequate, but it does not by itself establish that repair is better than trusting the better factorization.")
heading("6.1 Negative ablation: directional defect", 2)
para("A label-free heuristic was tested in which the factorization with the smaller marginalization defect receives greater trust. The heuristic does not reliably identify the factorization closer to truth as dependence strengthens; direction-selection accuracy falls toward or below chance in some regimes. This failure is retained as a negative result and should not be promoted as the main method.")
heading("6.2 Solver efficiency", 2)
para("The generalized Sinkhorn solver reaches the same soft objective as the independently implemented exponentiated-gradient path to numerical tolerance. In a local 5,000-sample, 5-by-5 microbenchmark, generalized scaling required about 0.022 seconds versus about 0.101 seconds for exponentiated gradient, roughly a 4.6x post-processing speedup. This is an implementation result, not a novelty claim over generalized Sinkhorn literature.")

heading("7 Real TFM validation - required before submission")
draft_box()
para("The central empirical blocker is real-model evidence. The final study will evaluate the same four-view extraction and reconciliation pipeline with TabPFN-3, TabICLv2, and TabFM. The first sanity test is Credit plus TabPFN-3, followed by the ten classification pairs used in the recent TFM consistency study and additional controlled OpenML extensions where feasible.")
table(["Dataset", "Rows", "A classes", "B classes"], [
    ("Anneal", "898", "5", "8"), ("Credit", "1,000", "2", "4"),
    ("Phishing", "1,353", "3", "3"), ("MIC", "1,596", "8", "2"),
    ("Customer", "1,723", "2", "6"), ("Car", "1,728", "4", "3"),
    ("Marketing", "2,216", "2", "8"), ("Wine", "6,497", "8", "2"),
    ("Nursery", "12,960", "5", "3"), ("Diamonds", "53,940", "5", "7"),
], [165, 80, 85, 85], 7.0)
para("The real-model section must report five-fold factorization and marginalization inconsistency, joint and conditional proper scores, calibration, consistency tax/dividend, runtime overhead, and paired statistical comparisons. Exact package and checkpoint versions plus model-weight licenses must be archived. No real-TFM performance claim is made in this draft.")

heading("8 Discussion")
para("The current controlled evidence suggests a more nuanced conclusion than a universal repair rule. Probabilistic coherence, calibration, and correctness are distinct properties. A coherent joint can be inferior to an incoherent factorization if the latter happens to align better with the true data-generating process. Conversely, strong dependence can create settings in which reconciling multiple views improves both joint and conditional scoring. The correct deployment question is therefore not simply whether a TFM is inconsistent, but whether the expected predictive dividend justifies the distortion needed to restore compatibility.")
para("Hard CoRe is attractive when direct marginals are trusted and downstream users require them to remain unchanged. Soft CoRe is better suited when marginal predictions can themselves be miscalibrated. Selective CoRe is the safest practical protocol because it allows validation data to retain an original factorization or a simple pool when repair would impose a tax.")
heading("8.1 Limitations", 2)
for item in [
    "Pairwise categorical setting only. Global coherence over many target variables introduces exponential state-space growth and requires additional structure.",
    "Regression and mixed categorical-continuous targets are deferred because continuous predictive heads require discretization or density approximations.",
    "Synthetic truth experiments are informative but cannot replace evaluation on released TFM checkpoints and real datasets.",
    "Validation selection consumes data and can overfit in very small samples; nested or cross-validated selection should be studied further.",
    "Post-hoc reconciliation does not repair the TFM training objective; a future training-time consistency loss or shared latent joint could be more principled.",
    "The originating TFM-consistency result is very recent, so novelty must be rechecked immediately before submission.",
]:
    bullet(item)

heading("9 Conclusion")
para("CoRe-TFM reframes probability consistency in tabular foundation models as a four-view reconciliation problem rather than a binary coherent-or-incoherent diagnosis. Hard CoRe performs a marginal-preserving information projection, Soft CoRe relaxes marginal fidelity through a convex objective with generalized Sinkhorn updates, and Selective CoRe permits validation evidence to reject repair when it would hurt prediction. Controlled known-truth experiments show that greater coherence does not automatically mean greater correctness and that the preferred strategy changes across dependence and data regimes. The remaining decisive step is the real-model benchmark on TabPFN-3, TabICLv2, and TabFM; until it is complete, this document remains a working draft rather than a submission-ready manuscript.")

heading("References - working list")
for ref in [
    "Klotergens et al. (2026). Do Tabular Foundation Models Agree with Themselves? Working preprint on marginalization and factorization consistency.",
    "Grinsztajn et al. (2026). TabPFN-3: Scaling Tabular Foundation Models.",
    "Qu et al. (2026). TabICLv2: Scalable Tabular In-Context Learning.",
    "Google Research (2026). Introducing TabFM: A Zero-Shot Foundation Model for Tabular Data.",
    "Mure (2017). Optimal compromise between incompatible conditional distributions.",
    "Benamou et al. (2015). Iterative Bregman projections for regularized transportation problems.",
    "Chizat et al. (2018). Scaling algorithms for unbalanced optimal transport problems.",
    "DistPFN authors (2026). Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment.",
    "Young (2026). Conditioning consistency in conditional neural processes.",
    "Additional calibration, density-estimation, and uncertainty references are maintained in the full working bibliography and will be normalized to final publication records before submission.",
]:
    para(ref, size=8.2, leading=9.6, after=2)

heading("Draft status and reproducibility")
para("Code: https://github.com/bnssaanirudh/core-tfm")
para("Current repository tests: 31 passing in the development environment. The public repository contains core probability construction, reconciliation methods, synthetic data generators, metrics, model adapters, tests, and reproduction notes. Real TFM checkpoints are intentionally not represented as completed evidence.")

footer()
c.save()

# ReportLab emits a four-byte binary marker after the PDF header. Replacing it
# with equal-length ASCII keeps xref byte offsets unchanged while making the
# generated working draft deterministic and easy to inspect.
data = open(OUT, "rb").read()
data = data.replace(b"%\x93\x8c\x8b\x9e", b"%ABCD", 1)
open(OUT, "wb").write(data)
print(f"Wrote {OUT} ({len(data)} bytes)")
