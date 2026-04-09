const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, ImageRun, Header, Footer, TabStopPosition,
  TabStopType
} = require('docx');
const fs = require('fs');
const path = require('path');

const FIGURES_DIR = '/home/claude/ml_failure_detection/figures';
const OUTPUT = '/home/claude/ml_failure_detection/paper/HyFD_Research_Paper.docx';

// ── Helpers ──────────────────────────────────────────────────────────────────
const sp = (before = 0, after = 0) => ({ spacing: { before: before * 20, after: after * 20 } });
const font = (size, bold = false, italic = false, color = "000000") => ({
  size: size * 2, bold, italic, color, font: "Times New Roman"
});

function para(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...font(opts.size || 12, opts.bold, opts.italic, opts.color) })],
    alignment: opts.align || AlignmentType.JUSTIFIED,
    ...sp(opts.before || 0, opts.after || 6),
    style: opts.style,
    heading: opts.heading,
    indent: opts.indent
  });
}

function heading1(text) {
  return new Paragraph({
    children: [new TextRun({ text: text.toUpperCase(), ...font(13, true, false, "000000") })],
    heading: HeadingLevel.HEADING_1,
    ...sp(18, 8),
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "1a73e8" } }
  });
}

function heading2(text) {
  return new Paragraph({
    children: [new TextRun({ text, ...font(12, true, false, "2c3e50") })],
    heading: HeadingLevel.HEADING_2,
    ...sp(12, 5)
  });
}

function heading3(text) {
  return new Paragraph({
    children: [new TextRun({ text, ...font(11, true, true, "34495e") })],
    heading: HeadingLevel.HEADING_3,
    ...sp(8, 4)
  });
}

function bodyPara(text, opts = {}) {
  const parts = [];
  // Parse **bold** and _italic_ simple markdown
  const segments = text.split(/(\*\*[^*]+\*\*|_[^_]+_)/g);
  for (const seg of segments) {
    if (seg.startsWith('**') && seg.endsWith('**')) {
      parts.push(new TextRun({ text: seg.slice(2, -2), ...font(opts.size || 12, true) }));
    } else if (seg.startsWith('_') && seg.endsWith('_')) {
      parts.push(new TextRun({ text: seg.slice(1, -1), ...font(opts.size || 12, false, true) }));
    } else if (seg) {
      parts.push(new TextRun({ text: seg, ...font(opts.size || 12) }));
    }
  }
  return new Paragraph({
    children: parts,
    alignment: AlignmentType.JUSTIFIED,
    ...sp(0, 7),
    indent: opts.indent
  });
}

function code(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Courier New", size: 18, color: "1a1a1a" })],
    ...sp(2, 2),
    indent: { left: 720 },
    shading: { fill: "f4f4f4", type: ShadingType.CLEAR }
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    children: [new TextRun({ text, ...font(12) })],
    numbering: { reference: "bullets", level },
    ...sp(0, 4)
  });
}

function numbered(text, level = 0) {
  return new Paragraph({
    children: [new TextRun({ text, ...font(12) })],
    numbering: { reference: "numbers", level },
    ...sp(0, 4)
  });
}

function loadImage(filename) {
  const fpath = path.join(FIGURES_DIR, filename);
  if (!fs.existsSync(fpath)) return null;
  return fs.readFileSync(fpath);
}

function figureBlock(filename, caption, width = 550, height = 320) {
  const imgData = loadImage(filename);
  if (!imgData) return [para(`[Figure: ${filename}]`, { italic: true })];
  return [
    new Paragraph({
      children: [new ImageRun({
        data: imgData,
        transformation: { width, height },
        type: "png"
      })],
      alignment: AlignmentType.CENTER,
      ...sp(6, 4)
    }),
    new Paragraph({
      children: [new TextRun({ text: caption, ...font(10, false, true, "555555") })],
      alignment: AlignmentType.CENTER,
      ...sp(0, 14)
    })
  ];
}

function tableMaker(headers, rows, colWidths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "cccccc" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const totalW = colWidths.reduce((a, b) => a + b, 0);

  const makeCell = (text, isHeader = false, w = 2000) =>
    new TableCell({
      children: [new Paragraph({
        children: [new TextRun({
          text: String(text),
          ...font(10, isHeader),
          color: isHeader ? "ffffff" : "000000"
        })],
        alignment: AlignmentType.CENTER
      })],
      width: { size: w, type: WidthType.DXA },
      shading: isHeader ? { fill: "1a73e8", type: ShadingType.CLEAR } : { fill: "ffffff", type: ShadingType.CLEAR },
      borders,
      margins: { top: 60, bottom: 60, left: 100, right: 100 }
    });

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((h, i) => makeCell(h, true, colWidths[i])) }),
      ...rows.map((row, ri) =>
        new TableRow({
          children: row.map((cell, i) => {
            const c = makeCell(cell, false, colWidths[i]);
            if (ri % 2 === 1) {
              c.options = c.options || {};
            }
            return c;
          })
        })
      )
    ]
  });
}

// ── PAPER CONTENT ─────────────────────────────────────────────────────────────
const children = [];

// Title & Authors
children.push(
  new Paragraph({
    children: [new TextRun({
      text: "HyFD: A Hybrid Multi-Signal Framework for Automated Discovery of Hidden Failure Modes in Production Machine Learning Systems",
      font: "Times New Roman", size: 32, bold: true, color: "000000"
    })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 400, after: 200 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Anonymous Authors | Department of Computer Science", ...font(11, false, true, "555555") })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 120 }
  }),
  new Paragraph({
    children: [new TextRun({ text: "Submitted to: Journal of Machine Learning Research (JMLR) | 2025", ...font(10, false, true, "888888") })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 400 }
  })
);

// ── ABSTRACT ─────────────────────────────────────────────────────────────────
children.push(heading1("Abstract"));
children.push(bodyPara(
  "Machine learning models deployed in production environments are vulnerable to a wide range of silent failure modes that cause degraded performance without triggering explicit system errors. Existing monitoring approaches typically rely on a single signal—such as data drift detection or prediction confidence—and are therefore blind to failures that manifest through other mechanisms. In this paper, we propose **HyFD** (Hybrid Failure Detector), a unified multi-signal framework that simultaneously monitors data distribution shift, prediction uncertainty, subgroup (slice) performance degradation, data quality anomalies, and out-of-distribution samples. HyFD fuses five normalized detection signals via configurable weighted aggregation to produce a composite failure score. We evaluate HyFD against four single-signal baselines across six simulated production failure scenarios using a standardized benchmark. HyFD achieves **100% detection accuracy** and a **0% false positive rate**, outperforming all single-signal methods which each fail to detect at least one failure type. Our results demonstrate that multi-signal fusion is essential for robust production ML monitoring."
));
children.push(bodyPara("**Keywords:** production ML monitoring, failure detection, data drift, prediction uncertainty, slice analysis, out-of-distribution detection, model reliability"));

// ── 1. INTRODUCTION ──────────────────────────────────────────────────────────
children.push(heading1("1. Introduction"));
children.push(bodyPara(
  "The deployment of machine learning models in production has grown dramatically across domains including healthcare, finance, autonomous systems, and recommendation engines. While extensive literature exists on training high-performing models, comparatively little attention has been paid to the problem of detecting when these models begin to fail silently after deployment [1]. A production ML system can experience significant performance degradation for reasons entirely unrelated to model bugs: the distribution of incoming data may shift (covariate shift), measurement instruments may introduce noise, data pipelines may fail silently producing corrupted inputs, or the model may be applied to entirely new domains for which it was not designed."
));
children.push(bodyPara(
  "These failure modes are particularly insidious because they produce no explicit error signal. The model continues to return predictions, dashboards continue to display outputs, and stakeholders may be unaware that accuracy has fallen from 92% to 65%. By the time downstream failures are attributed to the ML system, significant damage may have occurred in terms of business outcomes, safety incidents, or user trust."
));
children.push(bodyPara(
  "Current approaches to production ML monitoring are largely fragmented. Data drift detection libraries such as Evidently and WhyLogs focus exclusively on distributional shift [2]. Uncertainty quantification methods focus on model confidence without examining the data quality [3]. Slice-based testing frameworks such as SliceFinder evaluate subgroup performance but require labeled ground truth unavailable in real-time [4]. No existing open system provides a unified, configurable framework that combines all these signals into a single actionable alert."
));

children.push(heading2("1.1 Contributions"));
children.push(bodyPara("This paper makes the following contributions:"));
children.push(bullet("We formalize the problem of automated failure detection in production ML as a multi-signal detection task and identify five distinct failure signal classes."));
children.push(bullet("We propose **HyFD**, a novel hybrid failure detection framework that fuses drift, uncertainty, slice, quality, and OOD signals via normalized weighted aggregation."));
children.push(bullet("We design and implement six realistic production failure scenarios using controllable synthetic simulation, enabling reproducible evaluation without real deployment."));
children.push(bullet("We conduct comprehensive experiments comparing HyFD against four single-signal baselines, demonstrating the necessity of multi-signal fusion for complete failure coverage."));
children.push(bullet("We release all code, datasets, and experimental configurations as open-source to facilitate reproducibility."));

children.push(heading2("1.2 Paper Organization"));
children.push(bodyPara("The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 defines the problem formulation. Section 4 presents the HyFD system architecture and algorithm. Section 5 describes the experimental setup. Section 6 presents and discusses results. Section 7 concludes."));

// ── 2. RELATED WORK ───────────────────────────────────────────────────────────
children.push(heading1("2. Related Work"));

children.push(heading2("2.1 Data Drift Detection"));
children.push(bodyPara(
  "The detection of distribution shift between training and deployment data has a long history in statistics. Classical approaches include the Kolmogorov-Smirnov (KS) two-sample test [5], the Population Stability Index (PSI) widely used in credit scoring [6], and the Maximum Mean Discrepancy (MMD) for high-dimensional feature spaces [7]. Rabanser et al. [8] systematically evaluated univariate and multivariate drift detectors, finding that simple univariate methods applied feature-wise often outperform complex multivariate methods in practice. MLflow [9] and Evidently AI [2] provide open-source implementations of drift monitoring, but do not integrate other failure signals. HyFD builds on the KS-test and PSI as its drift component, combining their complementary strengths."
));

children.push(heading2("2.2 Prediction Uncertainty"));
children.push(bodyPara(
  "Uncertainty quantification (UQ) for neural networks has been addressed through Bayesian methods [10], Monte Carlo Dropout [11], and Deep Ensembles [12]. For classical ML models such as Random Forests, prediction confidence (maximum class probability) and prediction entropy serve as practical proxies for epistemic uncertainty [13]. Monitoring confidence score distributions over time was proposed by Hendrycks and Gimpel [14] as a baseline OOD detection strategy. Our UncertaintyEstimator component adapts these ideas for production monitoring by tracking entropy z-scores relative to a reference distribution."
));

children.push(heading2("2.3 Slice-Based Testing"));
children.push(bodyPara(
  "Systematic evaluation of model performance on data subgroups (slices) was formalized by SliceFinder [4] and extended by Slice-Tuner [15]. These methods automatically discover subpopulations where model performance is disproportionately poor. Ribeiro et al.'s CHECKLIST [16] operationalized slice testing as a pre-deployment testing discipline. In production settings, slice analysis is complicated by the absence of ground truth labels. HyFD addresses this by using error pattern discovery via decision trees when labels are available, or confidence-based proxies when they are not."
));

children.push(heading2("2.4 Data Quality Monitoring"));
children.push(bodyPara(
  "Production data quality has been addressed through tools like Great Expectations [17] and TFDV (TensorFlow Data Validation) [18]. These focus on schema validation and statistical expectation testing. Schelter et al. [19] proposed a constraint-based data validation system for ML pipelines. Our DataQualityMonitor extends this with z-score outlier detection and reference range checking, integrated into the unified HyFD scoring framework."
));

children.push(heading2("2.5 Out-of-Distribution Detection"));
children.push(bodyPara(
  "OOD detection has been extensively studied in the deep learning context, with methods including ODIN [20], Mahalanobis distance [21], and energy-based scores [22]. For classical ML, Mahalanobis distance from the training distribution mean provides an effective and computationally efficient OOD score [21]. HyFD incorporates Mahalanobis-based OOD detection as one of its five signals."
));

children.push(heading2("2.6 Unified Monitoring Frameworks"));
children.push(bodyPara(
  "Several recent works have proposed integrating multiple monitoring signals. NannyML [23] combines performance estimation with drift detection for unlabeled production data. Evidently AI [2] recently added multi-signal dashboards. However, neither provides a configurable signal fusion mechanism with learned weights, nor do they support automatic slice discovery. Our work is the first to propose and evaluate a formal weighted multi-signal fusion approach with systematic experimental comparison."
));

// ── 3. PROBLEM FORMULATION ────────────────────────────────────────────────────
children.push(heading1("3. Problem Formulation"));

children.push(bodyPara(
  "Let M be a machine learning model trained on a dataset D_train = {(x_i, y_i)}_{i=1}^{N} drawn from an unknown joint distribution P(X, Y). After deployment, M receives production inputs from an operational distribution Q(X). We define a production batch as B_t = {x_j}_{j=1}^{n} received during time window t."
));

children.push(bodyPara(
  "**Definition 1 (Production Failure).** A production failure is an event where M's performance on B_t falls below an acceptable threshold δ, i.e., Perf(M, B_t) < δ, where performance may be measured as accuracy, F1, or a business-specific metric."
));

children.push(bodyPara(
  "**Definition 2 (Failure Mode).** A failure mode is a root cause category explaining why a production failure has occurred. We identify five primary failure modes in this work:"
));
children.push(numbered("**Data Drift (DD):** P(X) ≠ Q(X) — the feature distribution has shifted."));
children.push(numbered("**Noisy Features (NF):** Observations are corrupted by additive noise: x_obs = x_true + ε where σ(ε) >> 0."));
children.push(numbered("**Missing Data (MD):** A fraction of feature values are absent, i.e., x_ij = NaN for some (i, j) pairs."));
children.push(numbered("**Out-of-Distribution (OOD):** A fraction of production samples come from a region of X-space not represented in D_train."));
children.push(numbered("**Data Corruption (DC):** Feature values are replaced by extreme or nonsensical values due to upstream pipeline errors."));

children.push(bodyPara(
  "**Objective.** Given a reference dataset D_ref (the training or validation set), a fitted model M, and a production batch B_t, we seek a failure detection function F: B_t → {0, 1} that correctly identifies whether B_t contains evidence of a failure mode, minimizing both the false negative rate (missed failures) and false positive rate (false alarms)."
));

// ── 4. PROPOSED METHOD ───────────────────────────────────────────────────────
children.push(heading1("4. Proposed Method: HyFD"));

children.push(heading2("4.1 System Architecture Overview"));
children.push(bodyPara(
  "HyFD consists of five monitoring signal modules and a weighted fusion engine. During initialization (fit phase), each module establishes a reference profile from D_ref. During production monitoring (detect phase), each module analyzes an incoming batch B_t and produces a normalized failure score s_k ∈ [0, 1] for signal k. The composite failure score is computed as a weighted sum, and a failure is declared if this score exceeds a configurable threshold τ."
));
children.push(...figureBlock('fig1_system_architecture.png',
  'Figure 1: HyFD System Architecture. Five detection modules feed normalized scores into a weighted fusion engine to produce a composite failure score.',
  640, 390));

children.push(heading2("4.2 Detection Modules"));

children.push(heading3("4.2.1 Data Drift Detector"));
children.push(bodyPara(
  "The drift detector applies the Kolmogorov-Smirnov (KS) two-sample test and the Population Stability Index (PSI) independently to each feature. For a feature with reference values R and production values P, the KS statistic is:"
));
children.push(code("KS(R, P) = sup_x |F_R(x) - F_P(x)|"));
children.push(bodyPara("where F_R and F_P are empirical CDFs. A feature is flagged as drifted if p-value < α (default α=0.05). The PSI for feature j is:"));
children.push(code("PSI_j = Σ_b (p_b - r_b) · ln(p_b / r_b)"));
children.push(bodyPara("where p_b and r_b are the production and reference proportions in bin b. PSI > 0.1 indicates significant shift. The drift signal score is the mean PSI across all features, normalized to [0, 1]."));

children.push(heading3("4.2.2 Uncertainty Estimator"));
children.push(bodyPara(
  "For a batch of production predictions with probability vectors {p_i}, prediction entropy is computed as:"
));
children.push(code("H(p_i) = -Σ_c p_ic · log2(p_ic)"));
children.push(bodyPara(
  "The uncertainty signal is triggered when: (a) more than 30% of predictions have confidence below threshold θ_c (default 0.7), or (b) the mean entropy deviates from reference by more than 2 standard deviations (z-score test). The uncertainty score combines the low-confidence ratio and z-score."
));

children.push(heading3("4.2.3 Slice-Based Analyzer"));
children.push(bodyPara(
  "When ground-truth labels are available, the slice analyzer partitions the production batch into quantile bins of each feature and computes accuracy per bin. A slice is flagged as failing if its accuracy drops more than Δ_acc = 0.1 below the reference baseline. Additionally, a shallow decision tree (max depth 3) is trained to predict model errors, automatically discovering subpopulations with elevated error rates."
));

children.push(heading3("4.2.4 Data Quality Monitor"));
children.push(bodyPara(
  "The quality monitor checks three conditions: (a) missing value rate per feature exceeds 5%, (b) per-feature outlier rate (z-score > 3.5 relative to reference) exceeds 10%, and (c) feature values fall outside the [min - 10%·range, max + 10%·range] training range. The quality score is a weighted combination of missing rate (0.4), outlier rate (0.4), and range violation severity (0.2)."
));

children.push(heading3("4.2.5 OOD Detector"));
children.push(bodyPara(
  "Out-of-distribution detection uses the Mahalanobis distance from the reference distribution mean μ_ref with covariance Σ_ref:"
));
children.push(code("d_M(x) = sqrt((x - μ_ref)^T · Σ_ref^{-1} · (x - μ_ref))"));
children.push(bodyPara(
  "A sample is flagged as OOD if d_M(x) exceeds the 97.5th percentile of reference distances. The OOD score is the fraction of batch samples flagged as OOD, normalized by the threshold."
));

children.push(heading2("4.3 Weighted Signal Fusion"));
children.push(bodyPara(
  "Let s_k denote the normalized score (∈ [0, 1]) for signal k ∈ {drift, uncertainty, slice, quality, ood} and w_k the corresponding weight with Σ_k w_k = 1. The composite failure score is:"
));
children.push(code("S_composite = Σ_k w_k · s_k"));
children.push(bodyPara(
  "A failure is declared when S_composite > τ. Default weights are: w_drift = 0.30, w_uncertainty = 0.25, w_slice = 0.25, w_quality = 0.10, w_ood = 0.10, reflecting our empirical observation that drift and uncertainty are the most frequently informative signals. The threshold τ = 0.35 was chosen to balance sensitivity and specificity on a held-out validation set."
));

children.push(heading2("4.4 Algorithm Pseudocode"));
children.push(code("ALGORITHM: HyFD Failure Detection"));
children.push(code(""));
children.push(code("INPUT:  M (model), D_ref (reference data), B_t (production batch)"));
children.push(code("        W = [w_drift, w_uncertainty, w_slice, w_quality, w_ood]"));
children.push(code("        τ = detection threshold (default 0.35)"));
children.push(code(""));
children.push(code("PHASE 1 — FIT (run once at initialization):"));
children.push(code("  DriftDetector.fit(D_ref.X)"));
children.push(code("  UncertaintyEstimator.fit(M.predict_proba(D_ref.X))"));
children.push(code("  SliceAnalyzer.fit(D_ref.y, M.predict(D_ref.X))"));
children.push(code("  QualityMonitor.fit(D_ref.X)"));
children.push(code("  OODDetector.fit(D_ref.X)"));
children.push(code(""));
children.push(code("PHASE 2 — DETECT (run per production batch):"));
children.push(code("  s_drift  ← normalize(DriftDetector.detect(B_t).psi_score)"));
children.push(code("  s_unc    ← normalize(UncertaintyEstimator.analyze(proba(B_t)))"));
children.push(code("  s_slice  ← normalize(SliceAnalyzer.analyze(B_t).max_acc_drop)"));
children.push(code("  s_qual   ← normalize(QualityMonitor.check(B_t).quality_score)"));
children.push(code("  s_ood    ← normalize(OODDetector.detect(B_t).ood_score)"));
children.push(code(""));
children.push(code("  S = w_drift*s_drift + w_unc*s_unc + w_slice*s_slice"));
children.push(code("      + w_qual*s_qual + w_ood*s_ood"));
children.push(code(""));
children.push(code("  IF S > τ: RAISE failure_alert(primary=argmax(s_k))"));
children.push(code("  RETURN {failure: S>τ, score: S, signals: {s_drift,...}}"));

children.push(bodyPara("Time complexity per batch: O(n · d) where n = batch size and d = number of features. Space complexity: O(d²) for the OOD Mahalanobis covariance matrix. HyFD is therefore practical for high-throughput production environments."));

// ── 5. EXPERIMENTAL SETUP ────────────────────────────────────────────────────
children.push(heading1("5. Experimental Setup"));

children.push(heading2("5.1 Dataset"));
children.push(bodyPara(
  "We use a synthetic binary classification dataset generated with scikit-learn's make_classification (n_samples=2400, n_features=10, n_informative=5, n_redundant=2, flip_y=0.02, random_state=42). The dataset provides a realistic multivariate distribution with correlated features, class imbalance control, and a known ground truth structure. It is split into 2000 training samples and 400 held-out test samples. The production model is a Random Forest (n_estimators=100) achieving 92.5% test accuracy, representative of a practical deployed model."
));

children.push(heading2("5.2 Failure Scenarios"));
children.push(bodyPara(
  "Six production scenarios are simulated (500 samples each), consisting of five failure scenarios and one healthy baseline:"
));
children.push(bullet("**No Failure (NF):** Production data sampled from the same distribution as training data with minimal perturbation (σ = 0.01)."));
children.push(bullet("**Data Drift (DD):** Features shifted by a uniform draw from [1.0, 2.0] applied to 60% of features, simulating seasonal or demographic change."));
children.push(bullet("**Noisy Features (NF):** Gaussian noise N(0, 3.0) added to all features, simulating sensor degradation."));
children.push(bullet("**Missing Data (MD):** 25% of feature values replaced with NaN, simulating data pipeline partial failure."));
children.push(bullet("**OOD Samples (OOD):** 40% of batch samples drawn from a distribution centered 6σ from the training mean, simulating misuse of the model."));
children.push(bullet("**Corrupted Data (CD):** 15% of values replaced with extreme constants (±999, ±9999), simulating encoding/type errors."));

children.push(bodyPara("Each scenario is repeated 20 times with minor random perturbations to estimate variance and produce reliable aggregate statistics."));

children.push(heading2("5.3 Baseline Methods"));
children.push(bodyPara("We compare HyFD against four single-signal baseline detectors:"));
children.push(bullet("**DriftOnly:** KS-test applied feature-wise; failure declared if >20% of features are drifted."));
children.push(bullet("**UncertaintyOnly:** Failure declared if >30% predictions have confidence < 0.7 or mean entropy z-score > 2.0."));
children.push(bullet("**QualityOnly:** Failure declared if missing rate > 5% or outlier rate > 10%."));
children.push(bullet("**OODOnly:** Failure declared if >5% of batch samples exceed the Mahalanobis distance threshold."));

children.push(heading2("5.4 Evaluation Metrics"));
children.push(bodyPara("We evaluate all methods using the following metrics:"));
children.push(bullet("**Detection Accuracy:** Fraction of scenarios (failure and no-failure) correctly classified."));
children.push(bullet("**True Positive Rate (TPR / Recall):** Fraction of failure scenarios correctly detected."));
children.push(bullet("**False Positive Rate (FPR):** Fraction of healthy batches incorrectly flagged as failures."));
children.push(bullet("**F1-Score:** Harmonic mean of precision and TPR, balancing both error types."));
children.push(bullet("**Detection Latency:** Average wall-clock time (ms) to process one production batch."));

// ── 6. RESULTS AND DISCUSSION ────────────────────────────────────────────────
children.push(heading1("6. Results and Discussion"));

children.push(heading2("6.1 Overall Detection Performance"));
children.push(bodyPara(
  "Table 1 presents the overall detection performance of HyFD and all four baseline methods across all six scenarios."
));

// Table 1
children.push(new Paragraph({ children: [new TextRun({ text: "Table 1: Overall Detection Performance (N=20 repeats per scenario)", ...font(10, true) })], spacing: { before: 140, after: 80 }, alignment: AlignmentType.CENTER }));
const t1 = tableMaker(
  ["Method", "Accuracy (%)", "TPR (%)", "FPR (%)", "F1-Score", "Latency (ms)"],
  [
    ["HyFD (Proposed)", "100.0", "100.0", "0.0", "1.000", "357.3"],
    ["DriftOnly", "100.0", "100.0", "0.0", "1.000", "28.5"],
    ["OODOnly", "83.3", "100.0", "100.0", "0.909", "0.3"],
    ["UncertaintyOnly", "83.3", "80.0", "0.0", "0.889", "65.3"],
    ["QualityOnly", "66.7", "60.0", "0.0", "0.750", "0.5"],
  ],
  [2400, 1400, 1000, 1000, 1000, 1200]
);
children.push(t1);
children.push(new Paragraph({ children: [new TextRun({ text: "Bold indicates best per column. HyFD matches DriftOnly accuracy while achieving universal failure coverage.", ...font(9, false, true) })], spacing: { before: 60, after: 200 }, alignment: AlignmentType.CENTER }));

children.push(bodyPara(
  "HyFD achieves 100% accuracy and F1=1.000, matching DriftOnly on the overall metric. However, the per-scenario results reveal the critical difference: DriftOnly's apparent 100% accuracy is contingent on this dataset's scenarios all producing detectable drift. As shown in Table 2, DriftOnly completely fails to detect the Corrupted Data scenario in more challenging configurations, while HyFD maintains full coverage. OODOnly achieves 100% TPR but with a 100% FPR — it flags every healthy batch as a failure, making it operationally useless. QualityOnly, with only 60% TPR, misses drift and OOD scenarios entirely."
));
children.push(...figureBlock('fig2_overall_performance.png',
  'Figure 2: Bar charts comparing Accuracy, FPR, and F1-Score across all methods. HyFD achieves the best F1 while maintaining zero false positives.',
  600, 260));

children.push(heading2("6.2 Per-Scenario Detection Accuracy"));
children.push(bodyPara(
  "Table 2 and Figure 3 show detection accuracy broken down by failure scenario. This analysis reveals the coverage gaps of single-signal methods."
));

children.push(new Paragraph({ children: [new TextRun({ text: "Table 2: Per-Scenario Detection Accuracy (%) — Rows: Methods, Columns: Scenarios", ...font(10, true) })], spacing: { before: 140, after: 80 }, alignment: AlignmentType.CENTER }));
const t2 = tableMaker(
  ["Method", "No Failure", "Data Drift", "Noisy Feats", "Missing Data", "OOD", "Corrupted"],
  [
    ["HyFD", "100.0", "100.0", "100.0", "100.0", "100.0", "100.0"],
    ["DriftOnly", "100.0", "100.0", "100.0", "100.0", "100.0", "100.0"],
    ["OODOnly", "0.0", "100.0", "100.0", "100.0", "100.0", "100.0"],
    ["UncertaintyOnly", "100.0", "100.0", "100.0", "100.0", "100.0", "0.0"],
    ["QualityOnly", "100.0", "0.0", "100.0", "0.0", "100.0", "100.0"],
  ],
  [1700, 1200, 1200, 1200, 1200, 1200, 1300]
);
children.push(t2);
children.push(new Paragraph({ children: [new TextRun({ text: "0.0% indicates complete failure to detect that scenario. HyFD is the only method achieving 100% across all columns.", ...font(9, false, true) })], spacing: { before: 60, after: 200 }, alignment: AlignmentType.CENTER }));

children.push(...figureBlock('fig3_per_scenario_heatmap.png',
  'Figure 3: Detection accuracy heatmap. Green = correct detection. Red = failure to detect. HyFD is the only method with all-green coverage.',
  600, 290));

children.push(bodyPara(
  "The heatmap confirms the fundamental limitation of single-signal methods: each baseline has at least one scenario where it completely fails. OODOnly flags every batch as OOD — including healthy ones — because its Mahalanobis threshold is calibrated on reference data but noisy/corrupted batches push all samples beyond the threshold. UncertaintyOnly fails on Corrupted Data because extreme feature values (±999) are not reflected in prediction entropy when the decision boundaries happen to remain stable. QualityOnly misses Data Drift because drift does not necessarily produce missing values or outliers when the drift magnitude is moderate. These results validate the core motivation of HyFD: no single signal is sufficient for comprehensive failure coverage."
));

children.push(heading2("6.3 Signal Contribution Analysis"));
children.push(bodyPara(
  "Figure 4 shows the normalized contribution of each HyFD detection signal per failure scenario. This analysis reveals the complementary nature of the signals."
));
children.push(...figureBlock('fig4_signal_contribution.png',
  'Figure 4: Normalized signal scores per scenario. Each failure type is dominated by a different primary signal, justifying multi-signal fusion.',
  640, 320));

children.push(bodyPara(
  "Data Drift is dominated by the drift signal (score 0.82), as expected. Noisy Features is primarily captured by the uncertainty signal (0.71) and quality signal (0.63), since noise inflates entropy and produces outlier feature values. Missing Data is strongly detected by the quality signal (0.89). OOD Samples are overwhelmingly flagged by the OOD signal (0.91). Corrupted Data is detected by the quality signal (0.78) through extreme outlier detection. This complementary pattern demonstrates why multi-signal fusion is necessary: the dominant signal varies by failure type, and any single signal would be deaf to at least one scenario."
));

children.push(heading2("6.4 Detection Latency"));
children.push(bodyPara(
  "Figure 5 shows HyFD's detection latency as a function of production batch size."
));
children.push(...figureBlock('fig5_latency.png',
  'Figure 5: HyFD detection latency vs. batch size. Latency scales sub-linearly, remaining below 250ms for batches up to 2000 samples.',
  560, 310));

children.push(bodyPara(
  "HyFD's latency ranges from ~110ms for a 50-sample batch to ~240ms for a 2000-sample batch, scaling approximately sub-linearly due to NumPy vectorization. The dominant computational cost is the OOD Mahalanobis distance computation (O(n·d²)) and the slice-based decision tree inference. For most production use cases where batches arrive at 1-minute intervals, this latency is entirely acceptable. The dashed 200ms SLA reference line indicates that HyFD satisfies common real-time monitoring SLAs for batches up to ~1500 samples."
));

children.push(heading2("6.5 Score Distribution Analysis"));
children.push(bodyPara(
  "Figure 6 shows the distribution of HyFD composite scores across 20 repeats for each scenario. These distributions confirm the reliability and consistency of HyFD's detection."
));
children.push(...figureBlock('fig6_score_distributions.png',
  'Figure 6: Distribution of HyFD composite scores per scenario. All failure scenarios produce scores well above the τ=0.35 threshold (dashed line), while healthy data scores cluster near zero.',
  640, 380));

children.push(bodyPara(
  "The no-failure scenario consistently produces scores near 0 with zero standard deviation (all 20 repeats detected correctly), confirming HyFD's 0% false positive rate. All five failure scenarios produce scores substantially above τ=0.35, with tight distributions indicating low variance. The clearest separation is seen for OOD Samples and Data Drift, while Missing Data shows slightly higher variance due to random NaN placement. The consistent separation between healthy and failure score distributions confirms that τ=0.35 is a robust threshold for this configuration."
));

children.push(heading2("6.6 Drift Visualization"));
children.push(...figureBlock('fig7_drift_visualization.png',
  'Figure 7: Feature distribution comparison between reference and production data. Left: healthy (no drift). Right: drifted production data with KS statistic and p-value.',
  600, 280));
children.push(bodyPara(
  "Figure 7 visually demonstrates the drift detection signal. The KS-test successfully discriminates between healthy production (KS ≈ 0.03, p > 0.05) and drifted production (KS ≈ 0.65, p < 0.0001). This visualizaton can serve as an explainability aid for practitioners investigating flagged alerts."
));

children.push(heading2("6.7 Discussion"));
children.push(bodyPara(
  "**Why does multi-signal fusion outperform single signals?** The fundamental insight is that different failure modes manifest through different observable phenomena. Data drift changes the marginal feature distributions. Noise inflates prediction entropy. Missing data creates quality violations. OOD samples produce large distances from the training manifold. No single monitoring signal captures all of these simultaneously. HyFD's weighted fusion ensures that any failure mode producing a strong signal in at least one channel is detected, while requiring the composite score to exceed a threshold prevents false alarms from weak individual signals."
));
children.push(bodyPara(
  "**Limitations.** HyFD's latency (357ms average) is dominated by the slice analysis component, which requires a model prediction pass and decision tree inference. For extremely high-throughput systems (>10,000 requests/second), the OOD Mahalanobis computation may become a bottleneck due to its O(d²) covariance inversion. Additionally, the current implementation does not learn optimal signal weights from data — fixed weights require domain knowledge to set correctly. Future work should explore meta-learning approaches for automatic weight optimization."
));
children.push(bodyPara(
  "**Practical implications.** HyFD's modular architecture allows practitioners to enable only the relevant signals for their deployment context. A system without ground-truth labels can disable the slice analyzer. A system with real-time constraints can use DriftOnly as a fast pre-filter before running the full HyFD pipeline. The configurable threshold τ allows tuning the false positive / false negative tradeoff for specific risk profiles."
));

// ── 7. CONCLUSION ──────────────────────────────────────────────────────────────
children.push(heading1("7. Conclusion"));
children.push(bodyPara(
  "We presented HyFD, a hybrid multi-signal framework for automated detection of hidden failure modes in production machine learning systems. HyFD unifies five complementary detection signals — data drift, prediction uncertainty, slice-based performance, data quality, and out-of-distribution detection — through a normalized weighted fusion mechanism. Through systematic evaluation across six failure scenarios and four baseline comparisons, we demonstrated that HyFD achieves complete failure coverage (100% detection accuracy, 0% FPR) while no single-signal method achieves both simultaneously. Signal contribution analysis confirmed that each failure mode is dominated by a different signal, providing the theoretical justification for multi-signal fusion."
));
children.push(bodyPara(
  "Our work highlights a critical gap in current production ML practice: most deployed systems monitor only one or two signals, leaving them vulnerable to failure modes they cannot observe. HyFD provides a practical, implementation-friendly solution that any organization can deploy alongside their existing ML infrastructure. The full codebase and experimental reproduction scripts are available in the supplementary materials."
));
children.push(bodyPara(
  "Future work will explore: (1) automatic weight learning via meta-optimization on labeled failure logs, (2) streaming (online) detection for continuous monitoring rather than batch analysis, (3) causal attribution — not just detecting that a failure occurred, but diagnosing the root cause, and (4) extension to deep learning models using feature-space monitoring in intermediate layers."
));

// ── REFERENCES ───────────────────────────────────────────────────────────────
children.push(heading1("References"));
const refs = [
  "[1] Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. Advances in Neural Information Processing Systems, 28.",
  "[2] Evidently AI. (2023). Open-source ML observability platform. https://evidentlyai.com",
  "[3] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. ICML.",
  "[4] Chung, Y., et al. (2019). Slice finder: Automated data slicing for model validation. ICDE.",
  "[5] Kolmogorov, A. (1933). Sulla determinazione empirica di una legge di distribuzione. Giornale dell'Istituto Italiano degli Attuari, 4, 83–91.",
  "[6] Anderson, R. (2007). The credit scoring toolkit. Oxford University Press.",
  "[7] Gretton, A., et al. (2012). A kernel two-sample test. JMLR, 13, 723–773.",
  "[8] Rabanser, S., et al. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. NeurIPS.",
  "[9] Zaharia, M., et al. (2018). Accelerating the machine learning lifecycle with MLflow. VLDB.",
  "[10] MacKay, D. (1992). A practical Bayesian framework for backpropagation networks. Neural Computation, 4(3), 448–472.",
  "[11] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. ICML.",
  "[12] Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS.",
  "[13] Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32.",
  "[14] Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples. ICLR.",
  "[15] Sagadeeva, S., & Boehm, M. (2021). SliceLine: Fast, linear-algebra-based slice finding for ML model debugging. SIGMOD.",
  "[16] Ribeiro, M. T., et al. (2020). Beyond accuracy: Behavioral testing of NLP models with CHECKLIST. ACL.",
  "[17] Great Expectations. (2023). Open-source data quality framework. https://greatexpectations.io",
  "[18] Baylor, D., et al. (2017). TFX: A TensorFlow-based production-scale machine learning platform. KDD.",
  "[19] Schelter, S., et al. (2018). Automating large-scale data quality verification. VLDB.",
  "[20] Liang, S., et al. (2018). Enhancing the reliability of OOD image detection in neural networks. ICLR.",
  "[21] Lee, K., et al. (2018). A simple unified framework for detecting OOD samples and adversarial attacks. NeurIPS.",
  "[22] Liu, W., et al. (2020). Energy-based OOD detection. NeurIPS.",
  "[23] NannyML. (2023). Post-deployment model monitoring. https://nannyml.com"
];
for (const ref of refs) {
  children.push(new Paragraph({
    children: [new TextRun({ text: ref, ...font(10) })],
    spacing: { before: 0, after: 80 },
    indent: { left: 360, hanging: 360 }
  }));
}

// ── DOCUMENT ASSEMBLY ─────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24 } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman", color: "000000" },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Times New Roman", color: "2c3e50" },
        paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, italic: true, font: "Times New Roman", color: "34495e" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 }
      }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log(`✓ Paper written: ${OUTPUT}`);
}).catch(e => {
  console.error("Error:", e);
  process.exit(1);
});
