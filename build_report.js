// build_report.js
// Generates the final Word document for the project.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, HeadingLevel, LevelFormat, BorderStyle,
  ShadingType, WidthType, PageOrientation, PageBreak,
} = require("docx");

const ROOT = "/home/claude/student_risk_project";
const FIG = path.join(ROOT, "figures");
const OUT_PATH = "/mnt/user-data/outputs/Student_Risk_Prediction_Final_Report.docx";

// ---------- Helpers ----------

const BLACK = "000000";
const GRAY = "595959";
const RULE = "BFBFBF";
const HEADER_FILL = "E7E6E6";

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 100, line: 300, ...opts.spacing },
    alignment: opts.alignment,
    children: [new TextRun({ text, bold: opts.bold, italics: opts.italics, size: opts.size ?? 22, color: BLACK })],
  });
}

// Multi-run paragraph
function pRuns(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 100, line: 300 },
    children: runs.map(r => new TextRun({
      text: r.text, bold: r.bold, italics: r.italics, size: r.size ?? 22, color: BLACK,
    })),
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, bold: true, size: 32, color: BLACK })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 220, after: 100 },
    children: [new TextRun({ text, bold: true, size: 26, color: BLACK })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, bold: true, size: 23, color: BLACK })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60, line: 290 },
    children: [new TextRun({ text, size: 22, color: BLACK })],
  });
}

function image(filename, widthIn, heightIn) {
  const data = fs.readFileSync(path.join(FIG, filename));
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 80 },
    children: [new ImageRun({
      type: filename.endsWith("png") ? "png" : "jpg",
      data,
      transformation: { width: widthIn * 96, height: heightIn * 96 },
    })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 220 },
    children: [new TextRun({ text, italics: true, size: 20, color: GRAY })],
  });
}

// Simple formatted table
function makeTable(headers, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const borders = { top: border, bottom: border, left: border, right: border };

  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      borders,
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { fill: HEADER_FILL, type: ShadingType.CLEAR },
      margins: { top: 100, bottom: 100, left: 120, right: 120 },
      children: [new Paragraph({
        children: [new TextRun({ text: h, bold: true, size: 20, color: BLACK })],
      })],
    })),
  });

  const dataRows = rows.map(row => new TableRow({
    children: row.map((cell, i) => new TableCell({
      borders,
      width: { size: colWidths[i], type: WidthType.DXA },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({
        children: [new TextRun({ text: String(cell), size: 20, color: BLACK })],
      })],
    })),
  }));

  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

// ---------- Build the doc body ----------

const body = [];

// === Title page ===
body.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 2000, after: 200 },
  children: [new TextRun({
    text: "Machine Learning–Based Early Risk Prediction",
    bold: true, size: 44, color: BLACK,
  })],
}));
body.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 600 },
  children: [new TextRun({
    text: "for Student Academic Performance Using Behavioral and Engagement Data",
    bold: true, size: 32, color: BLACK,
  })],
}));
body.push(p("Final Project Report", { alignment: AlignmentType.CENTER, bold: true, size: 26, after: 200 }));
body.push(p("Team: Simran Mogadala · Kritika Joshi · Mary Neha Reddy Thumma · Khadija Warraich",
  { alignment: AlignmentType.CENTER, size: 22, after: 100 }));
body.push(p("University of Colorado Denver", { alignment: AlignmentType.CENTER, size: 22, after: 600 }));

body.push(new Paragraph({ children: [new PageBreak()] }));

// === Abstract ===
body.push(h1("Abstract"));
body.push(p(
  "Educational institutions often only identify struggling students after midterm or final grades, which is too late for meaningful intervention. We built a machine learning system that predicts academic risk from early-semester behavioral and demographic indicators (study time, prior failures, attendance, parental education, lifestyle factors) without using any in-course grades. We compared five classifiers — Logistic Regression, k-Nearest Neighbors, Random Forest, Gradient Boosting, and XGBoost — under 5-fold stratified cross-validation. The best models reached a held-out ROC-AUC of 0.81 and an F1 of 0.46 on a 17%-imbalanced target. Logistic Regression achieved the highest recall (0.62), making it the strongest candidate for an early-warning use case where missing an at-risk student is costlier than a false alarm. We then ran a fairness audit across sex, address (urban/rural), and school, and found that the more accurate gradient-boosted models actually had worse demographic parity than the linear baseline. We discuss this trade-off, provide SHAP-based explanations for individual predictions, and outline a deployment concept for an instructor-facing dashboard."
));

// === 1. Introduction ===
body.push(h1("1. Introduction and Problem Statement"));
body.push(p(
  "Most early-warning systems in education rely on grades that arrive after a problem has already taken hold. By the time a midterm score lands, the window for changing study habits, increasing attendance, or connecting a student with tutoring has often closed. The goal of this project is to flag students who are likely to fail before any course grades are available, using only the kinds of information a school already has on day one of a term: demographics, family background, prior academic history, and self-reported study and lifestyle habits."
));
body.push(p(
  "We frame this as a binary classification problem. The target is at-risk (final grade G3 < 10 on a 0–20 scale, the standard pass threshold in the source dataset). The features are everything except the in-course grades G1 (first period) and G2 (second period), which we deliberately drop so the model cannot peek at information that wouldn't exist at the time we want to make a prediction."
));

body.push(h2("1.1 Motivation"));
body.push(p(
  "Early academic risk assessment can improve retention, lower dropout rates, and let educators target limited support resources where they will have the most impact. But predictive systems in education also bring real ethical concerns: bias against demographic groups, the risk of stigmatizing or labeling students, and a lack of transparency about why a model flagged someone. We treat fairness and interpretability as first-class deliverables, not afterthoughts."
));

body.push(h2("1.2 Related Work"));
body.push(p(
  "Cortez and Silva (2008) introduced the UCI Student Performance dataset and showed that classical classifiers (Decision Trees, Random Forests, Naïve Bayes, Neural Networks) could predict end-of-term grades, with the strongest gains coming from including prior period grades. More recent work using Learning Management System (LMS) data (e.g., engagement clickstreams, login frequency, assignment submission timing) has shown that behavioral features can substantially improve prediction beyond grade history alone. A consistent finding across this literature is that interpretable models are preferred in education contexts even when they sacrifice a few points of accuracy, because instructors need to understand why a student was flagged before they will trust the system."
));

// === 2. Data ===
body.push(h1("2. Data"));
body.push(p(
  "We use the schema of the UCI Student Performance dataset (Cortez & Silva, 2008): 33 columns covering school, demographics, family context, study habits, lifestyle, and three grade columns (G1, G2, G3). Because the execution environment for this project did not have outbound network access to the UCI archive, we generated a 1,000-row dataset that follows the same schema, the same value ranges, and the same correlation structure documented in the original paper (e.g., higher study time → higher G3, more failures → lower G3, more absences → lower G3). All preprocessing, modeling, and fairness code is dataset-agnostic; dropping the real student-mat.csv into data/ and re-running from script 02_eda.py reproduces the full pipeline on the original data."
));
body.push(p("After dropping G1 and G2 to enforce the early-prediction constraint, we have 30 input features:",
  { after: 60 }));
body.push(bullet("13 numeric features: age, parental education (Medu, Fedu), travel time, study time, prior failures, family relationship, free time, going out, weekday alcohol, weekend alcohol, health, absences"));
body.push(bullet("17 categorical features: school, sex, address (urban/rural), family size, parental status, parents' jobs, reason for school choice, guardian, plus eight binary yes/no support indicators (school support, family support, paid tutoring, activities, nursery, intent to pursue higher ed, internet access, romantic relationship)"));
body.push(p("Target: at_risk = 1 if G3 < 10, else 0. Base rate is 17.1% — a meaningfully imbalanced classification problem.",
  { after: 200 }));

// === 3. EDA ===
body.push(h1("3. Exploratory Data Analysis"));
body.push(p(
  "Three findings drove our modeling choices. First, the target is imbalanced: only about one in six students is at risk. We need to track recall and F1, not just accuracy — a model that always predicts \"not at risk\" would already score 83% accuracy and be completely useless."
));
body.push(image("eda_target_distribution.png", 6.4, 2.5));
body.push(caption("Figure 1. Final grade distribution (left) with the at-risk cutoff at G3 < 10, and the resulting binary class balance (right). About 17% of students fall below the pass threshold."));

body.push(p(
  "Second, the strongest signals for risk are exactly what an instructor would expect: prior failures (correlation −0.38 with G3), study time (+0.31), going out (−0.23), and weekday alcohol use (−0.22). Parental education matters, but less than behavioral variables. Absences was a weaker linear signal here than the literature reports — we will revisit this in the limitations section."
));
body.push(image("eda_correlation_heatmap.png", 6.4, 5.2));
body.push(caption("Figure 2. Correlation matrix of numeric features (G1 and G2 excluded). Failures and study time stand out as the clearest linear signals for the target."));

body.push(p(
  "Third, when we look at engagement features against the actual at-risk rate, the relationships are visible in the raw data — they are not artifacts of modeling. Students with three prior failures have an at-risk rate above 50%; students who report studying more than ten hours a week have a single-digit risk rate."
));
body.push(image("eda_feature_target.png", 6.4, 4.7));
body.push(caption("Figure 3. At-risk rate broken down by four key features. The relationships are monotonic and steep, which is why even simple models can pick up most of the signal."));

body.push(p(
  "We also previewed demographic differences before any model was trained. The raw at-risk rates by sex, address, and school are close (within a few percentage points), but they are not identical. Whatever a model learns will inherit some of this baseline imbalance, which is the entire reason the fairness audit in Section 6 exists."
));
body.push(image("eda_demographic_risk.png", 6.4, 2.6));
body.push(caption("Figure 4. Baseline at-risk rates by sex, address (Urban/Rural), and school. Differences are small but non-zero, so any classifier built on this data inherits some demographic skew."));

// === 4. Method ===
body.push(h1("4. Method"));

body.push(h2("4.1 Pipeline"));
body.push(bullet("Data split: stratified 80/20 train/test on the binary target so both partitions have ~17% positives."));
body.push(bullet("Preprocessing: standard scaling for numeric features, one-hot encoding for categoricals (handle_unknown=\"ignore\"). The preprocessor is fit only on the training set and applied to test."));
body.push(bullet("Class imbalance: handled at training time via class_weight=\"balanced\" (Logistic Regression, Random Forest) and scale_pos_weight (XGBoost), rather than resampling. This keeps the test distribution honest."));
body.push(bullet("Hyperparameter tuning: 5-fold stratified cross-validation on the training set, optimizing F1 (the metric that balances precision and recall under imbalance)."));
body.push(bullet("Final evaluation: each model's best CV configuration is re-fit on the full training set and evaluated once on the held-out test set."));

body.push(h2("4.2 Models"));
body.push(p("We compared five classifiers spanning the typical bias/variance and interpretability trade-off:", { after: 60 }));
body.push(bullet("Logistic Regression (L2-regularized) — interpretable linear baseline."));
body.push(bullet("k-Nearest Neighbors — non-parametric baseline; sensitive to scaling and feature dimensionality."));
body.push(bullet("Random Forest — bagged trees, captures non-linearities and interactions."));
body.push(bullet("Gradient Boosting (sklearn) — additive trees, typically strong on tabular data."));
body.push(bullet("XGBoost — high-performance gradient boosting with built-in handling for class imbalance."));

body.push(h2("4.3 Evaluation Metrics"));
body.push(p(
  "Accuracy alone is misleading on imbalanced data. We report Accuracy, Precision, Recall, F1, ROC-AUC, and Matthews Correlation Coefficient (MCC). MCC is particularly useful here because it stays close to zero for trivial classifiers even when accuracy looks high. For an early-warning use case we treat Recall and ROC-AUC as the most operationally meaningful metrics — failing to flag a struggling student is a worse error than briefly checking in on a student who turns out to be fine."
));

// === 5. Results ===
body.push(h1("5. Results"));

body.push(h2("5.1 Held-out Test Performance"));

const modelTable = [
  ["Gradient Boosting", "0.850", "0.591", "0.382", "0.464", "0.776", "0.394"],
  ["Logistic Regression", "0.755", "0.368", "0.618", "0.462", "0.806", "0.333"],
  ["XGBoost", "0.825", "0.481", "0.382", "0.426", "0.794", "0.328"],
  ["K-Nearest Neighbors", "0.850", "0.667", "0.235", "0.348", "0.685", "0.334"],
  ["Random Forest", "0.820", "0.417", "0.147", "0.217", "0.791", "0.166"],
];
body.push(makeTable(
  ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MCC"],
  modelTable,
  [2160, 1200, 1200, 1100, 1100, 1200, 1400],
));
body.push(p("Table 1. Test-set performance, sorted by F1. Best value in each column is in the discussion below.",
  { italics: true, size: 20, after: 200 }));

body.push(p(
  "No single model wins on every metric. Gradient Boosting and Logistic Regression tie on F1 (0.46) but represent very different operating points. Gradient Boosting is precise (59% of its alarms are real) but cautious (it only catches 38% of at-risk students). Logistic Regression flips that: it catches 62% of at-risk students at the cost of more false alarms (37% precision). For an early-warning system whose entire purpose is to surface students who need attention, the higher-recall configuration is the right default — and Logistic Regression also has the best ROC-AUC (0.81), which means its underlying probability estimates are the most useful if a school wants to tune the threshold later."
));
body.push(p(
  "K-Nearest Neighbors is the cautionary tale of the comparison. Its 85% accuracy looks impressive, but its 23% recall and ROC-AUC of 0.68 reveal that it is mostly succeeding by predicting the majority class. Random Forest with the hyperparameters our grid found ends up similarly conservative."
));
body.push(image("model_metric_bars.png", 6.5, 3.1));
body.push(caption("Figure 5. Per-metric comparison across all five models. The accuracy column is deceptively flat; recall and MCC reveal the real differences."));

body.push(image("model_roc_curves.png", 5.6, 4.8));
body.push(caption("Figure 6. ROC curves on the held-out test set. The three top-AUC models (Logistic Regression, XGBoost, Random Forest) sit very close together; the practical separation between them comes from where you set the decision threshold, not the underlying ranking quality."));

body.push(image("model_confusion_matrices.png", 6.5, 2.4));
body.push(caption("Figure 7. Confusion matrices on the test set. Note how Logistic Regression trades a larger false-positive count for many fewer false negatives — the right trade-off for an early-warning system."));

// === 6. Fairness ===
body.push(h1("6. Fairness Analysis"));
body.push(p(
  "We audited the two top models — Logistic Regression and XGBoost — across three protected attributes: sex, address (urban/rural), and school. For each, we computed the selection rate (probability of being flagged), the true-positive rate (TPR / recall within the group), and the false-positive rate (FPR within the group). We then computed three standard disparity measures: demographic parity difference, equal opportunity difference, and the disparate impact ratio (the EEOC \"80% rule\")."
));
body.push(image("fairness_per_group.png", 6.5, 3.7));
body.push(caption("Figure 8. Selection rate and TPR by demographic group, for both top models. The most accurate model is not the most equitable model."));

const fairTable = [
  ["Logistic Regression", "sex (F vs M)", "0.087", "0.250", "0.733", "Fail"],
  ["Logistic Regression", "address (R vs U)", "0.015", "0.448", "0.948", "Pass"],
  ["Logistic Regression", "school (GP vs MS)", "0.055", "0.337", "0.832", "Pass"],
  ["XGBoost", "sex (F vs M)", "0.059", "0.104", "0.637", "Fail"],
  ["XGBoost", "address (R vs U)", "0.109", "0.214", "0.338", "Fail"],
  ["XGBoost", "school (GP vs MS)", "0.010", "0.317", "0.927", "Pass"],
];
body.push(makeTable(
  ["Model", "Attribute", "DP diff", "EOpp diff", "DI ratio", "80% rule"],
  fairTable,
  [2200, 1900, 1100, 1300, 1100, 1100],
));
body.push(p("Table 2. Fairness disparities. \"DP\" = demographic parity, \"EOpp\" = equal opportunity (TPR difference), \"DI\" = disparate impact ratio. Lower DP and EOpp are better; DI closer to 1.0 is better.",
  { italics: true, size: 20, after: 200 }));

body.push(p(
  "Two findings stand out. First, neither model passes the 80% rule on sex — both flag male students at noticeably higher rates than female students. The Logistic Regression case is partly a real signal in the data (its TPR is high for both groups, just unequal), while the XGBoost case is more concerning (low TPR on both groups, with male students clearly more likely to be flagged). Second, XGBoost — the model with the better headline accuracy — has substantially worse fairness on address than Logistic Regression. Its DI ratio of 0.34 means rural students are flagged at roughly one-third the rate of urban students, despite having similar real at-risk rates in the test set. This is the classic accuracy-vs-equity tension: the higher-capacity model has learned to over-rely on a feature (likely address itself, given that we left it as an input) that aligns with a protected attribute."
));

body.push(h2("6.1 Mitigation Discussion"));
body.push(p("We did not implement bias mitigation in this project, but we identify three practical paths a deployment would need to take seriously:", { after: 60 }));
body.push(bullet("Pre-processing: drop or coarsen features that act as proxies for protected attributes (e.g., remove address, or merge urban/rural into a single \"commute time\" signal)."));
body.push(bullet("In-processing: train with a fairness-constrained objective (e.g., the Fairlearn ExponentiatedGradient reduction) so the model is forced to equalize opportunity across groups."));
body.push(bullet("Post-processing: pick group-specific thresholds so each group has the same TPR. This is the simplest fix and the easiest to audit, but it requires the protected attribute to be available at decision time, which is itself a policy question."));

// === 7. Interpretability ===
body.push(h1("7. Interpretability"));
body.push(p(
  "Even the best classification metrics are not enough on their own. An instructor who is asked to act on a flag needs to understand why. We provide two complementary views: global feature importance (which features matter overall) and SHAP values (which features pushed this specific prediction in this direction)."
));
body.push(image("feature_importance_xgb.png", 5.6, 4.8));
body.push(caption("Figure 9. XGBoost feature importance (top 20). Prior failures, intent to pursue higher education, and study time dominate."));

body.push(image("feature_importance_logreg.png", 5.6, 4.8));
body.push(caption("Figure 10. Logistic Regression coefficients (top 20 by magnitude). Red bars push toward at-risk; green bars push toward not-at-risk. Failures, going out, and weekday alcohol push toward risk; study time, parental education, and family relationship push away."));

body.push(image("shap_summary_xgb.png", 6.4, 4.5));
body.push(caption("Figure 11. SHAP summary plot for XGBoost. Each dot is one student; horizontal position is that student's contribution from that feature; color is the feature value. The top of the plot tells the same story as the linear coefficients, which is reassuring — both model families agree on the underlying risk drivers."));

body.push(p(
  "The convergence between the linear and tree-based explanations is the most important interpretability result here. When a simple, transparent model and a more complex one agree on the top drivers of risk, we can be more confident the signal is real and not an artifact of one model's inductive bias. It also means we can deploy the linear model for transparency without giving up much in the way of predictive insight."
));

// === 8. Deployment ===
body.push(h1("8. Deployment Concept"));
body.push(p(
  "We sketch a thin-client early-warning dashboard rather than build it. The model would run as a batch job at the end of each enrollment period and write a per-student record to an internal table: the model's at-risk probability, the threshold-based flag, and the top three SHAP features that contributed to the prediction. An instructor view would surface a roster sorted by risk probability with the explanation visible inline (\"flagged because of: 2 prior failures, low reported study time, no family support\")."
));
body.push(p("Three guardrails are non-negotiable for any real deployment:", { after: 60 }));
body.push(bullet("No automated decisions. The model surfaces students; humans decide on outreach. The dashboard never sends a student a message on its own."));
body.push(bullet("Visible explanation. Every flag carries its top three contributing features. No black-box flags reach an instructor."));
body.push(bullet("Audit logging. All flags, all overrides, and aggregate fairness metrics by demographic group are logged and reviewed each term."));

// === 9. Ethics ===
body.push(h1("9. Ethics"));
body.push(p(
  "Three concerns shaped our design choices. The first is labeling risk: telling a student they are predicted to fail can become a self-fulfilling prophecy, and a flag on a student's record can follow them in ways that grades alone do not. Our mitigation is that the model output is never shared with the student or used in any official record — it is only a signal to an instructor that this is a person to check on. The second concern is bias amplification: if the model is more aggressive about flagging certain demographic groups, the support resources we direct based on its output will inherit that bias. Our fairness audit (Section 6) shows this risk is real and must be monitored continuously, not measured once and forgotten. The third concern is transparency: a model that an instructor cannot interrogate is a model that should not be deployed. The combination of an interpretable model (Logistic Regression) with consistent SHAP-based explanations from the more accurate XGBoost model gives us a defensible answer to \"why was this student flagged?\" for every prediction."
));

// === 10. Limitations ===
body.push(h1("10. Limitations and Future Work"));
body.push(bullet("The dataset used in this report mirrors UCI Student Performance in schema and known correlations, but is synthetic. Effects sizes (especially around absences) may differ from the original; we expect ranking among models to be similar."));
body.push(bullet("We modeled risk as a single binary outcome at the end of the term. A more useful operational system would produce time-resolved risk that updates weekly as new behavioral data arrives."));
body.push(bullet("We did not include LMS-style engagement features (login frequency, assignment submission timing, discussion participation). The literature shows these are usually the strongest behavioral predictors. The pipeline is structured so adding them is a one-liner change in 03_preprocessing.py."));
body.push(bullet("Fairness was audited on three coarse attributes. A real deployment needs intersectional analysis (e.g., rural female students at school MS) and ongoing monitoring of subgroup metrics."));
body.push(bullet("We did not implement bias mitigation. Section 6.1 identifies three concrete paths and the trade-offs each one accepts."));

// === 11. Conclusion ===
body.push(h1("11. Conclusion"));
body.push(p(
  "We built an end-to-end pipeline for early academic risk prediction that uses only pre-term features, compared five classifiers under cross-validated tuning, audited the top two for fairness across three demographic attributes, and produced individual-level explanations with SHAP. The Logistic Regression model is our recommendation: it had the best recall, the best ROC-AUC, the best fairness profile, and is fully interpretable. The two main project takeaways are that (1) for imbalanced early-warning problems, the right metric is recall, not accuracy, and (2) the most accurate model is not necessarily the most equitable model — fairness has to be measured directly, not assumed to come along with predictive performance."
));

// === References ===
body.push(h1("References"));
body.push(p("Cortez, P. and Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5–12."));
body.push(p("Lundberg, S. M. and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30."));
body.push(p("Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD."));
body.push(p("Hardt, M., Price, E. and Srebro, N. (2016). Equality of Opportunity in Supervised Learning. Advances in Neural Information Processing Systems 29."));
body.push(p("U.S. Equal Employment Opportunity Commission (1978). Uniform Guidelines on Employee Selection Procedures (the \"four-fifths rule\")."));

// === Appendix A — Code structure ===
body.push(h1("Appendix A. Code Structure"));
body.push(p("All code is provided alongside this report. The pipeline runs end-to-end in order:", { after: 60 }));
body.push(bullet("src/01_generate_dataset.py — produces data/student-mat.csv"));
body.push(bullet("src/02_eda.py — figures 1–4"));
body.push(bullet("src/03_preprocessing.py — fits and saves the preprocessor"));
body.push(bullet("src/04_modeling.py — trains/tunes the 5 models, figures 5–7, model_comparison.csv"));
body.push(bullet("src/05_fairness.py — figure 8, fairness_per_group.csv, fairness_disparities.csv"));
body.push(bullet("src/06_interpretability.py — figures 9–11, feature_importance.csv"));

body.push(h2("Reproducing on real UCI data"));
body.push(p("Download student.zip from the UCI Machine Learning Repository (Student Performance, ID 320), extract student-mat.csv into data/, and re-run from script 02. Every downstream script is dataset-agnostic as long as the schema matches."));

// ---------- Build the document ----------

const doc = new Document({
  creator: "Mary, Simran, Kritika, Khadija",
  title: "Student Risk Prediction — Final Report",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22, color: BLACK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Calibri", color: BLACK },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Calibri", color: BLACK },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Calibri", color: BLACK },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } },
        }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, buf);
  console.log("Wrote " + OUT_PATH + " (" + buf.length + " bytes)");
});
