# A Calibrated Machine Learning Framework for Plasma Proteomic Prediction
# of Acute Tubular Injury: Methodological Development and Proof-of-Concept
# in the Kidney Precision Medicine Project

**Julian Borges, MD**
Frontier Translational Research Lab, Department of Computer Science,
Boston University, Boston, MA, USA

**Correspondence:** jyborges@bu.edu | ORCID: 0009-0001-9929-3135
**Running title:** Calibrated Proteomic ML Framework for ATI
**Reporting standard:** TRIPOD+AI — Collins GS et al. *BMJ* 2024;385:e078378
**Code:** https://github.com/julian-borges-md/ati-proteomics (MIT licence)
**Data DOI:** https://doi.org/10.48698/6VRN-ZE53 (open access, no DUA required)

**Related prior publication disclosure:** The KPMP SomaScan datasets used here
were previously analysed by Schmidt IM et al. (*Nat Commun.* 2024;15:7368) for
biomarker discovery. The present paper addresses a distinct scientific objective
(calibrated prediction modelling) and was conducted independently using publicly
available data. See cover letter for full disclosure.

---

## Abstract

**Background:** Biomarker discovery studies have identified plasma proteins
associated with acute tubular injury (ATI) severity using the SomaScan platform.
A methodological gap remains: no study has developed a calibrated ML prediction
framework generating individual-level ATI probability estimates optimised for
clinical decision-making.

**Methods:** We developed a two-arm calibrated ML pipeline applied to harmonized
KPMP SomaScan plasma datasets (2022, n=136; 2024, n=184; combined n=320, ATI n=28;
DOI: 10.48698/6VRN-ZE53). Datasets were harmonized on 7,481 shared aptamers with
batch covariate. Arm 1: logistic regression on clinical variables. Arm 2: XGBoost
on top-500 variance-selected proteins (selected within cross-validation folds) plus
clinical variables, with Platt calibration. Both arms validated by stratified 5-fold
cross-validation. Primary analysis: AKI-enrolled subgroup (n=61, ATI n=28, ATI-negative
n=33) — the only clinically valid comparison. Secondary: full cohort. Metrics: AUROC,
AUPRC, Brier Skill Score, ECE. Bootstrap 95% CIs (n=2000) and label-permutation tests
(n=1000) reported. Reported per TRIPOD+AI.

**Results:** Primary AKI-only analysis: Clinical LR AUROC 0.611 [0.461–0.746],
AUPRC 0.609 [0.436–0.767]; XGBoost AUROC 0.520 [0.370–0.669], AUPRC 0.544
[0.375–0.712]. Both span chance; ATI is not detectable at this sample size within
AKI. Secondary full-cohort analysis: Clinical LR AUROC 0.939 [0.900–0.969],
XGBoost AUROC 0.896 [0.837–0.947] — reflecting AKI vs non-AKI enrollment
separation, not ATI signal. XGBoost Brier Skill Score 0.230 vs −0.029 for Clinical LR,
demonstrating superior probability calibration. No significant AUROC difference between
models (bootstrap difference p=0.115).

**Conclusions:** Within AKI, ATI is not detectable from plasma proteomics at current
sample sizes. The pipeline infrastructure — calibrated XGBoost, proper cross-validation,
and Brier Skill Score evaluation — provides a methodological template for future
powered studies. External validation with ATI n≥100 is required.

**Keywords:** acute tubular injury, plasma proteomics, SomaScan, machine learning,
XGBoost, probability calibration, KPMP, TRIPOD+AI

---

## 1. Introduction

Acute tubular injury is the most common histopathological finding across nearly
all forms of kidney disease and a principal mediator of the transition from AKI to
CKD.[1,2] Its non-invasive diagnosis has long been a clinical priority, motivating
substantial plasma and urinary biomarker research.[3,4]

The diagnostic toolkit for ATI remains inadequate. Serum creatinine and eGFR
rise only after substantial nephron loss and cannot localise injury to the tubular
compartment.[5,6] Urinary microscopy is operator-dependent and poorly standardised.[7]
Kidney biopsy is the gold standard but invasive and not serialisable.[1]

High-dimensional plasma proteomics using SomaScan has substantially advanced
biomarker discovery in nephrology.[8,9] Schmidt and colleagues identified 156
proteins associated with ATI severity at Bonferroni-corrected thresholds in 434
biopsy-confirmed patients, with replication in three cohorts including the KPMP.[10]
That work defines the current state of knowledge: a rich candidate biomarker set.

However, biomarker association and clinical prediction are distinct problems.
Identifying proteins that correlate with ATI severity does not establish whether
those proteins can generate well-calibrated individual-level probability estimates.
Calibration — the alignment between predicted probabilities and observed event rates —
determines clinical operationalisability.[11] To date, no study has developed a
calibrated ML prediction framework for ATI from plasma SomaScan proteomics.

This paper addresses that gap. We develop a two-arm calibrated ML pipeline applied
to harmonized KPMP SomaScan Plasma datasets (n=320, DOI: 10.48698/6VRN-ZE53) using
Platt sigmoid calibration as the central methodological contribution. The primary
analysis is restricted to AKI-enrolled participants — the only scientifically valid
comparison for ATI detection, since all ATI-positive participants were AKI-enrolled.

Specific aims: (1) evaluate whether plasma proteomic features, combined with clinical
variables, can detect ATI within AKI; (2) compare Platt-calibrated versus uncalibrated
probability output using Brier Skill Score and ECE; (3) test for significant AUROC
difference between clinical-only and proteomic+clinical models; (4) provide a
reproducible pipeline for future larger-scale proteomic ATI prediction studies.

---

## 2. Methods

### 2.1 Data Source and Study Population

Two KPMP SomaScan plasma datasets were accessed via open access (no DUA required)
at https://atlas.kpmp.org:

- **2022 release:** SS-2214310, SomaScan v4.0, ANML-normalised, n=136,
  accessed April 12, 2026.
- **2024 release:** SS-2495199 v4.1, ANML-SMP-normalised, n=184,
  accessed April 12, 2026 (DOI: 10.48698/6VRN-ZE53).

Both comprise biopsy-confirmed kidney disease participants from the KPMP
multisite prospective cohort.[12] The releases contain entirely distinct
participants (zero overlap confirmed by Participant ID). Combined cohort: n=320.

ATI status was derived from the KPMP Open Access Clinical Data file
(20260309_OpenAccessClinicalData.csv). Participants with Primary Adjudicated
Category "Acute Tubular Injury" were ATI-positive (n=28); all others ATI-negative
(n=292). All 28 ATI-positive participants were AKI-enrolled (28/28, 100%).

### 2.2 Dataset Harmonization

The 2022 (v4.0) and 2024 (v4.1) releases differ by 238 aptamers: 7,604 measured
in 2022, 7,596 in 2024, with 7,481 shared. Analysis was restricted to the 7,481
common aptamers. Protein abundances were log2-transformed (clipped at 1 RFU). A
binary batch covariate (0=2022, 1=2024) was included in all models. No explicit
batch correction was applied; batch covariate inclusion is sufficient for
tree-based models. Pooling is justified by: (1) identical KPMP protocol and
biopsy adjudication; (2) same EDTA plasma ANML normalisation strategy; (3) 98.4%
and 98.5% panel overlap respectively; (4) batch covariate explicitly models
residual inter-release technical variation.[13]

### 2.3 Feature Engineering

Clinical features: baseline eGFR (mL/min/1.73 m², median imputation for missing),
KDIGO Stage (ordinal: Stage 1=1, Stage 2=2, Stage 3=3; missing coded as 0,
justified because missing KDIGO indicates non-AKI enrollment where KDIGO is
not applicable, making 0 a meaningful category distinct from Stage 1–3),
batch covariate. Proteomic features: top-500 SomaScan aptamers by variance.

**Important:** Variance-based feature selection was performed inside each
cross-validation fold on the training partition only (n≈256). This prevents
any information leakage from test folds into the feature selection step.
The selected 500 proteins differ slightly across folds; reported features
are from the full-data selection for interpretability only.


### 2.4 Model Development

**Arm 1 — Clinical baseline:** Logistic regression (L2, C=1.0,
class_weight='balanced', max_iter=1000, StandardScaler) on eGFR, KDIGO Stage,
and batch covariate.

**Arm 2 — Combined proteomic+clinical (primary model):** XGBoost
(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.1, scale_pos_weight=10.4) on top-500 variance-selected
proteins plus eGFR, KDIGO Stage, and batch covariate. scale_pos_weight=10.4
is the training-fold negative-to-positive ratio (292/28), addressing the
10.4:1 class imbalance. Platt sigmoid calibration applied via
CalibratedClassifierCV(method='sigmoid', cv=3) within each outer fold.

### 2.5 Validation Protocol

Stratified 5-fold cross-validation (StratifiedKFold, n_splits=5, shuffle=True,
random_state=42). Feature selection (variance ranking) performed inside each
training fold on the training partition only, preventing test-set information
leakage. All out-of-fold predictions collected into a single prediction vector
for metric computation.

### 2.6 Performance Metrics and Statistical Tests

**Primary analysis:** AKI-enrolled subgroup (n=61: 28 ATI-positive, 33
ATI-negative). This is the only clinically valid comparison: the hypothesis
that plasma proteomics can detect ATI requires comparing ATI-positive and
ATI-negative participants within the clinical context where ATI occurs (AKI).

**Secondary analysis:** Full harmonized cohort (n=320). Reported to
demonstrate the enrichment bias but not as the primary performance claim.

**Metrics:** AUROC and AUPRC (co-primary). AUPRC is reported because at 8.8%
prevalence (full cohort) and 45.9% prevalence (AKI-only), AUROC can be
optimistic; AUPRC reflects performance at the operational class distribution.
Random classifier AUPRC baselines: full cohort 0.088, AKI-only 0.459.

**Brier Skill Score (BSS):** BSS = 1 − Brier/Brier_naive, where
Brier_naive = prevalence × (1 − prevalence). BSS > 0 indicates improvement
over a naive classifier that predicts base rate for all observations. BSS is
the primary calibration adequacy metric as it evaluates both discrimination
and calibration jointly relative to a meaningful baseline.

**Expected Calibration Error (ECE):** 5-bin quantile method.
ECE = mean(|fraction_positive_in_bin − mean_predicted_in_bin|). Interpreted
qualitatively given limited ATI events per bin (~5–6 positives total).

**Uncertainty quantification:** Bootstrap 95% CIs on AUROC and AUPRC from
2000 resamples (numpy default_rng seed=42). Resamples lacking both classes
excluded.

**Model comparison:** Bootstrap difference test on AUROC (n=2000 resamples),
two-sided p-value = 2 × min(P(diff ≥ 0), P(diff ≤ 0)). Tests whether
proteomic+clinical model significantly outperforms clinical-only baseline.

**Sensitivity analysis — permutation test:** Labels shuffled 1000 times;
AUROC recomputed on fixed OOF predictions. p-value = (count null ≥ observed + 1)
/ 1001. This tests prediction-label correlation, not model refit permutation,
and is interpreted as a basic non-randomness check rather than a strong
inferential test.

### 2.7 Feature Importance

XGBoost gain-based importance computed on full training set (n=320) with
full-dataset variance selection for interpretability. Feature rankings from
within-CV selection differ slightly per fold; full-data ranks are reported
as descriptive summaries, not inferential findings.

### 2.8 Reproducibility

Python 3.14, scikit-learn ≥1.4, XGBoost ≥2.0. All random states fixed
(random_state=42). Full pipeline and unit tests at
https://github.com/julian-borges-md/ati-proteomics (MIT). GitHub Actions CI
active. Reported per TRIPOD+AI.[14]

---

## 3. Results

### 3.1 Cohort Characteristics

The harmonized cohort comprised 320 participants (136 from 2022, 184 from 2024).
ATI-positive: n=28 (8.8%); ATI-negative: n=292 (91.2%). Full characteristics
are in Table 1.

All 28 ATI-positive participants were AKI-enrolled (28/28, 100%); ATI-negative
participants included CKD (n=210, 71.9%), AKI (n=33, 11.3%), Healthy Reference
(n=32, 11.0%), and DM-R (n=17, 5.8%). This composition defines the study
population as a biopsy-enriched cohort; the AKI-enrolled subgroup (n=61) is
the pre-specified primary analysis.

Baseline eGFR was higher in ATI-positive participants (mean 81.9 ± 22.4) than
ATI-negative (62.0 ± 25.9; p<0.001, two-sample t-test), reflecting acute injury
on a higher pre-morbid baseline. KDIGO Stage distribution differed markedly
(p<0.001): Stage 3 in 21/28 (75%) ATI-positive vs 14/292 (4.8%) ATI-negative,
consistent with severe biopsy-indicated AKI. Diabetes was less frequent in
ATI-positive (6/28, 21.4% vs 183/292, 62.7%; p<0.001). Hypertension differed
significantly (p<0.001). Sex and batch did not differ significantly (p=0.135
and p=0.066 respectively).

**Table 1. Cohort Characteristics by ATI Status**

| Characteristic | ATI-Positive (n=28) | ATI-Negative (n=292) | Total (n=320) | p-value |
|---|---|---|---|---|
| **Batch** | | | | |
| 2022 release | 17 (60.7%) | 119 (40.8%) | 136 (42.5%) | 0.066 |
| 2024 release | 11 (39.3%) | 173 (59.2%) | 184 (57.5%) | |
| **Enrollment Category** | | | | |
| AKI | 28 (100.0%) | 33 (11.3%) | 61 (19.1%) | <0.001 |
| CKD | 0 (0.0%) | 210 (71.9%) | 210 (65.6%) | |
| Healthy Reference | 0 (0.0%) | 32 (11.0%) | 32 (10.0%) | |
| DM-R | 0 (0.0%) | 17 (5.8%) | 17 (5.3%) | |
| **Sex** | | | | |
| Male | 20 (71.4%) | 160 (54.8%) | 180 (56.3%) | 0.135 |
| Female | 8 (28.6%) | 132 (45.2%) | 140 (43.8%) | |
| **Baseline eGFR (mL/min/1.73 m²)** | | | | |
| Mean ± SD | 81.9 ± 22.4 | 62.0 ± 25.9 | 63.8 ± 26.2 | <0.001 |
| Median | 76.5 | 55.2 | 57.3 | |
| **KDIGO Stage** | | | | |
| Stage 1 | 2 (7.1%) | 5 (1.7%) | 7 (2.2%) | <0.001 |
| Stage 2 | 3 (10.7%) | 11 (3.8%) | 14 (4.4%) | |
| Stage 3 | 21 (75.0%) | 14 (4.8%) | 35 (10.9%) | |
| Not applicable/missing | 2 (7.1%) | 262 (89.7%) | 264 (82.5%) | |
| **Diabetes History** | | | | |
| Yes | 6 (21.4%) | 183 (62.7%) | 189 (59.1%) | <0.001 |
| No | 20 (71.4%) | 107 (36.6%) | 127 (39.7%) | |
| Don't know/missing | 2 (7.1%) | 2 (0.7%) | 4 (1.3%) | |
| **Hypertension History** | | | | |
| Yes | 11 (39.3%) | 230 (78.8%) | 241 (75.3%) | <0.001 |
| No | 17 (60.7%) | 59 (20.2%) | 76 (23.8%) | |
| Don't know/missing | 0 (0.0%) | 3 (1.0%) | 3 (0.9%) | |

*p-values: continuous variables from two-sample t-test; categorical from chi-squared.
ATI: Acute Tubular Injury. AKI: Acute Kidney Injury. CKD: Chronic Kidney Disease.
DM-R: Diabetes Mellitus Resilience. eGFR: estimated glomerular filtration rate.*


### 3.2 Primary Analysis: ATI Detection Within AKI

The primary analysis was restricted to AKI-enrolled participants (n=61: 28
ATI-positive, 33 ATI-negative). Results are in Table 2 and Figure 1B.

Clinical LR AUROC was 0.611 [0.461–0.746] with AUPRC 0.609 [0.436–0.767].
XGBoost calibrated AUROC was 0.520 [0.370–0.669] with AUPRC 0.544 [0.375–0.712].
Both models' AUROC confidence intervals span 0.5 (chance). Both Brier Skill
Scores were negative (Clinical LR BSS −0.507; XGBoost BSS −0.258), indicating
both models perform worse than a naive classifier that predicts base rate for
all participants in the AKI subgroup. The AUROC difference between models was
not significant (bootstrap difference p=0.230).

**This is the principal finding.** ATI is not detectable from plasma proteomics
within the AKI-enrolled population at this sample size. All ATI-positive
participants were AKI-enrolled, meaning the clinically meaningful question —
can we identify which AKI patients have ATI — cannot be answered with n=28.

The AKI-only random classifier AUPRC baseline is 0.459 (equal to AKI-only
ATI prevalence 28/61). Clinical LR AUPRC of 0.609 marginally exceeds this
baseline, suggesting clinical variables (eGFR, KDIGO Stage) provide modest
discriminative information for ATI within AKI. However, given the CI width
([0.436–0.767] spanning the baseline), this should not be over-interpreted.

### 3.3 Secondary Analysis: Full-Cohort Performance

Full-cohort results (n=320, ATI n=28, 8.8% prevalence) are in Table 2
and Figure 1A. These results are reported for transparency but should not
be interpreted as evidence of ATI diagnostic performance.

Clinical LR achieved AUROC 0.939 [0.900–0.969] and AUPRC 0.588 [0.407–0.750].
XGBoost calibrated achieved AUROC 0.896 [0.837–0.947] and AUPRC 0.493
[0.310–0.665]. The AUROC difference between models was not significant
(bootstrap difference p=0.115, 95% CI of difference includes 0).

These high AUROCs reflect the separation between AKI-enrolled ATI-positive
participants and the predominantly CKD/Reference ATI-negative participants,
not ATI detection within a clinically homogeneous population. eGFR and KDIGO
Stage — both strongly associated with AKI enrollment — drive this discrimination.

### 3.4 Probability Calibration and Brier Skill Scores

Figure 2 presents calibration curves for all three models. Calibration results
reveal important differences in probability quality.

Clinical LR ECE was 0.135, indicating poor calibration — predicted probabilities
deviate 13.5 percentage points from observed rates on average. BSS was −0.029,
meaning Clinical LR provides no probability skill over predicting the base rate.
Despite AUROC 0.939, it produces systematically miscalibrated probabilities.

XGBoost uncalibrated ECE was 0.018 with BSS 0.258, indicating well-calibrated
predictions with 25.8% improvement over naive baseline — the best calibration
of the three models. XGBoost Platt-calibrated ECE was 0.036 with BSS 0.230.
In this dataset, Platt calibration modestly worsened ECE relative to uncalibrated
XGBoost, though BSS remained positive and above naive. This unexpected result
reflects that tree ensemble outputs from XGBoost can be naturally well-calibrated
when scale_pos_weight is appropriately set; Platt calibration adds no benefit
and may reduce sharpness when the base model is already calibrated.

ECE estimates should be interpreted qualitatively. With 5 quantile bins and
approximately 5–6 ATI-positive cases total, each bin contains 1–2 positive
events; bin-level calibration estimates are inherently unstable.

**Table 2. Model Performance — Stratified 5-Fold Cross-Validation**

| Analysis | Model | AUROC (95% CI) | AUPRC (95% CI) | Brier | BSS | ECE |
|---|---|---|---|---|---|---|
| **Primary: AKI-only** | | | | | | |
| (n=61, ATI+=28) | Clinical LR | 0.611 [0.461–0.746] | 0.609 [0.436–0.767] | 0.374 | −0.507 | — |
| | XGBoost+Platt | 0.520 [0.370–0.669] | 0.544 [0.375–0.712] | 0.312 | −0.258 | — |
| | AUROC diff p-value | p=0.230 | | | | |
| **Secondary: Full cohort** | | | | | | |
| (n=320, ATI+=28) | Clinical LR | 0.939 [0.900–0.969] | 0.588 [0.407–0.750] | 0.082 | −0.029 | 0.135 |
| | XGBoost uncalibrated | 0.925 [0.887–0.957] | 0.493 [0.319–0.681] | 0.059 | 0.258 | 0.018 |
| | XGBoost+Platt | 0.896 [0.837–0.947] | 0.493 [0.310–0.665] | 0.061 | 0.230 | 0.036 |
| | AUROC diff (LR vs XGB+Platt) p-value | p=0.115 | | | | |
| | Permutation p (both) | p=0.001 | | | | |

*95% CIs from bootstrap resampling (n=2000). BSS: Brier Skill Score vs naive baseline
(prevalence × (1−prevalence); full cohort naive=0.080, AKI-only naive=0.248).
BSS>0 indicates improvement over naive classifier. Random AUPRC: full cohort 0.088,
AKI-only 0.459. ECE: Expected Calibration Error, 5-bin quantile (qualitative only
given n=28). Permutation test: label-shuffle on fixed OOF predictions.*


### 3.5 Feature Importance

Figure 3A presents the top-20 XGBoost gain-based predictors from the full-dataset
model. KDIGO Stage ranked first, reinforcing the dominance of clinical variables.
PRDX3 (Peroxiredoxin-3, mitochondrial antioxidant) ranked second, biologically
plausible given mitochondrial oxidative stress in tubular injury.[15] PPFIA1
(Liprin-alpha-1, cell adhesion), DLX4 (transcription factor), and NPLOC4
(ubiquitin pathway) followed. These rankings reflect multivariate predictive
contribution in this specific dataset; they should not be interpreted as
biomarker validation.

### 3.6 Batch Effects

The 2022 cohort contributed 17/28 (60.7%) ATI-positive and 119/292 (40.8%)
ATI-negative participants. ATI prevalence differed by batch (12.5% vs 6.0%),
consistent with enrollment composition variation across release waves. Batch
did not differ significantly between ATI groups (p=0.066). Batch covariate
did not rank in the top-20 XGBoost predictors, suggesting batch membership
did not dominate proteomic signal.

---

## 4. Discussion

### 4.1 Principal Finding

ATI is not detectable from plasma SomaScan proteomics within the AKI-enrolled
population at n=28. Both models achieved near-chance AKI-only AUROC (0.611 and
0.520), with CIs spanning chance and negative Brier Skill Scores indicating
worse-than-naive probability predictions. This is the honest, primary scientific
finding of this paper.

This result does not indicate that ATI has no plasma proteomic signature. Schmidt
et al. 2024 demonstrated clear associations between 156 proteins and ATI
severity.[10] It indicates that 28 ATI-positive cases are insufficient to train
a supervised classifier to detect ATI within the subset of participants where
it clinically occurs (AKI). The biological signal exists; the sample size is
inadequate to operationalise it as a prediction model.

### 4.2 The Full-Cohort AUROC and Why It Is Misleading

The secondary full-cohort analysis yielded AUROC 0.939 and 0.896. These numbers
are not evidence of diagnostic performance. They reflect the classifier learning
to distinguish AKI-enrolled participants from CKD, DM-R, and Reference participants
— a task that KDIGO Stage and eGFR largely solve. The Clinical LR BSS of −0.029
confirms this: despite AUROC 0.939, the clinical LR provides no probability skill
over predicting base rate for all participants. A model that perfectly discriminates
between groups can still fail to provide useful individual-level probabilities.

This distinction — between AUROC (ranking accuracy) and BSS (probability skill)
— is a methodological contribution of this paper. Future proteomic prediction
studies should report both metrics rather than AUROC alone.

### 4.3 Calibration Findings

XGBoost uncalibrated achieved the best calibration (ECE 0.018, BSS 0.258). Platt
calibration marginally degraded ECE (0.018 to 0.036) while maintaining positive
BSS (0.230). This unexpected result suggests that properly weighted XGBoost
(scale_pos_weight set to the negative-to-positive ratio) produces naturally
calibrated outputs in this class-imbalanced setting. Platt calibration may be
redundant when scale_pos_weight is correctly specified. Future studies should
evaluate uncalibrated XGBoost alongside Platt calibration rather than assuming
calibration always helps.

### 4.4 Relationship to Schmidt et al. 2024

This paper builds on, not competes with, Schmidt et al. 2024.[10] That study
answered which proteins associate with ATI severity; this paper answers whether
those proteins support a supervised classifier. The negative primary finding here
does not contradict Schmidt et al. — it establishes the minimum sample size
requirement for the next translational step.

### 4.5 Limitations

(1) **Sample size is the dominant limitation.** ATI n=28 produces ~5–6 positive
cases per CV fold. AUROC estimates have standard errors of ~0.15–0.20 per fold;
all metrics are exploratory. The pre-registered primary analysis (AKI-only) shows
near-chance performance; this should drive the field toward powered studies.

(2) **Cohort enrichment.** The biopsy-enriched KPMP cohort is not representative
of clinical populations where ATI must be identified among unselected AKI patients.

(3) **Batch harmonization.** Panel version differences (v4.0 vs v4.1) and
normalisation differences (ANML vs ANML-SMP) may introduce residual technical
variation not fully captured by the batch covariate.

(4) **No external validation.** All results from internal 5-fold CV on a single
harmonized dataset.

(5) **Permutation test limitation.** The implemented permutation test shuffles
labels on fixed OOF predictions, not a full model-refit permutation. It tests
prediction-label correlation (p=0.001 is expected given the strong eGFR signal)
but is weaker than a proper model-refit permutation test. Interpretation is
limited to confirming non-random predictions.

(6) **ECE instability.** Five quantile bins with ~5–6 ATI-positive cases total
yields 1–2 positives per bin. ECE values should be treated as qualitative.

(7) **Cross-sectional design.** Diagnostic model only; temporal or prognostic
inference not supported.

### 4.6 Future Directions

The minimum requirement for a meaningful ATI prediction study is ATI n≥100
(providing 80% power to detect AUROC 0.75 vs 0.5 at alpha=0.05). Controlled-access
KPMP data (n>800 with DUA) is the most accessible near-term path. This pipeline
— with inside-CV feature selection, BSS reporting, and the primary AKI-only
analysis structure — is ready to be applied to that larger dataset without
methodological modification.

---

## 5. Conclusion

This study establishes that ATI is not detectable from plasma SomaScan proteomics
in the AKI-enrolled population at the current open-access sample size (ATI n=28),
with both models producing near-chance AKI-only AUROC and negative Brier Skill
Scores within AKI. The full-cohort AUROC of 0.939/0.896 reflects enrollment
category separation, not ATI signal, and is accompanied by negative BSS for the
clinical model. The methodological contributions — proper inside-CV feature
selection, BSS as a calibration metric, bootstrap CIs, and AKI-only primary
analysis framing — provide a rigorous template for future powered ATI prediction
studies. The pipeline is openly released at
https://github.com/julian-borges-md/ati-proteomics (MIT licence).

---

## Data Availability

KPMP SomaScan plasma datasets available via open access at https://atlas.kpmp.org
(DOI: 10.48698/6VRN-ZE53; accessed April 12, 2026). Clinical data: same repository
(20260309_OpenAccessClinicalData.csv). Code:
https://github.com/julian-borges-md/ati-proteomics (MIT licence).

## KPMP Data Citation

The results here are in whole or part based upon data generated by the Kidney
Precision Medicine Project. Accessed April 12, 2026. https://www.kpmp.org.
The KPMP is supported by NIDDK through grants: U01DK133081, U01DK133091,
U01DK133092, U01DK133093, U01DK133095, U01DK133097, U01DK114866, U01DK114908,
U01DK133090, U01DK133113, U01DK133766, U01DK133768, U01DK114907, U01DK114920,
U01DK114923, U01DK114933, U24DK114886, UH3DK114926, UH3DK114861, UH3DK114915,
and UH3DK114937.

## References

1. Wen Y, et al. *Kidney Int Rep.* 2020;5:1993. doi:10.1016/j.ekir.2020.08.026
2. Muiru AN, et al. *Ann Intern Med.* 2023;176:961. doi:10.7326/M22-3617
3. Parikh CR, et al. *Kidney Int.* 2006;70:199. doi:10.1038/sj.ki.5001527
4. Coca SG, et al. *J Am Soc Nephrol.* 2014;25:1063. doi:10.1681/ASN.2013070757
5. Waikar SS, et al. *J Am Soc Nephrol.* 2012;23:13. doi:10.1681/ASN.2010111124
6. Bonventre JV, Yang L. *J Clin Invest.* 2011;121:4210. doi:10.1172/JCI45161
7. Perazella MA, Coca SG. *Clin J Am Soc Nephrol.* 2012;7:167. doi:10.2215/CJN.09490911
8. Schlosser P, et al. *J Am Soc Nephrol.* 2021;32:2355. doi:10.1681/ASN.2021040478
9. Moledina DG, et al. *J Am Soc Nephrol.* 2021;32:1424. doi:10.1681/ASN.2020091330
10. Schmidt IM, et al. *Nat Commun.* 2024;15:7368. doi:10.1038/s41467-024-51304-x
11. Van Calster B, et al. *BMC Med.* 2019;17:230. doi:10.1186/s12916-019-1466-7
12. KPMP Consortium. *JCI Insight.* 2022;7:e154882. doi:10.1172/jci.insight.154882
13. Gold L, et al. *PLoS One.* 2010;5:e15004. doi:10.1371/journal.pone.0015004
14. Collins GS, et al. *BMJ.* 2024;385:e078378. doi:10.1136/bmj-2023-078378
15. Lee S, et al. *Redox Biol.* 2018;17:121. doi:10.1016/j.redox.2018.04.013
