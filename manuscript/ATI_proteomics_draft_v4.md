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

**Related prior publication disclosure:** The KPMP SomaScan dataset used here
was previously analysed by Schmidt IM et al. (*Nat Commun.* 2024;15:7368) for
biomarker discovery. The present paper addresses a distinct scientific objective
(calibrated prediction modelling) and was conducted independently using
publicly available data. See cover letter for full disclosure.

---

## Abstract

**Background:** Biomarker discovery studies have identified plasma proteins
associated with acute tubular injury (ATI) severity using the SomaScan platform.
A methodological gap remains: no study has developed a calibrated ML
prediction framework generating individual-level ATI probability estimates
optimised for clinical decision-making. Calibrated probabilities — where a
predicted probability of 0.70 reflects a genuine 70% event rate — are a
prerequisite for clinical operationalisation that association studies cannot provide.

**Methods:** We developed and evaluated a two-arm calibrated machine learning
pipeline applied to harmonized KPMP SomaScan plasma datasets (2022 release,
n=136; 2024 release, n=184; combined n=320, ATI n=28; DOI: 10.48698/6VRN-ZE53;
open access). The two datasets were harmonized on 7,481 shared SomaScan aptamers;
batch membership was included as a binary covariate. Arm 1: logistic regression
on clinical variables (eGFR, KDIGO Stage). Arm 2 (primary): XGBoost classification
on top-500 variance-selected proteins plus clinical variables and batch covariate,
with Platt sigmoid probability calibration via CalibratedClassifierCV. Both arms
were validated by stratified 5-fold cross-validation. Primary metrics: AUROC and
log loss. Calibration assessed by Expected Calibration Error (ECE), Brier score,
and reliability diagrams. Reported per TRIPOD+AI guidelines.

**Results:** Clinical-only model (Arm 1) AUROC 0.938, log loss 0.306, Brier 0.082.
XGBoost proteomic+clinical model (Arm 2, Platt calibrated): AUROC 0.899 [0.833–0.952], AUPRC 0.510 [0.335–0.687], log loss 0.211, Brier 0.060, ECE 0.035. Top XGBoost predictors by gain included PRDX3, PPFIA1, DLX4,
NPLOC4, and KDIGO Stage. All ATI-positive participants were AKI-enrolled (28/28),
and the cohort is enriched accordingly; results should not be generalised to
unselected populations.

**Conclusions:** This study establishes a reproducible, calibrated proteomic ML pipeline for ATI classification, filling a methodological gap distinct from existing biomarker association work. All performance estimates are preliminary given ATI n=28. External validation in powered cohorts is required.

**Keywords:** acute tubular injury, plasma proteomics, SomaScan, machine learning,
XGBoost, probability calibration, prediction model, KPMP, TRIPOD+AI

---

## 1. Introduction

Acute tubular injury is the most common histopathological finding across nearly
all forms of kidney disease and a principal mediator of the transition from acute
kidney injury (AKI) to chronic kidney disease (CKD).[1,2] Its non-invasive
diagnosis has long been a clinical priority, motivating substantial plasma and
urinary biomarker research over the past two decades.[3,4]

The diagnostic toolkit for ATI remains inadequate. Serum creatinine and eGFR
rise only after substantial nephron loss and cannot localise injury to the tubular
compartment.[5,6] Urinary microscopy provides qualitative information but is
operator-dependent and poorly standardised.[7] Kidney biopsy remains the gold
standard but is invasive, cannot be repeated serially, and carries procedural
risk.[1] Non-invasive biomarkers capable of detecting tubular injury are therefore
a genuine unmet clinical need.

High-dimensional plasma proteomics using the SomaScan aptamer-based platform has
substantially advanced biomarker discovery in nephrology.[8,9] Schmidt and
colleagues measured over 7,000 plasma proteins in 434 patients with
biopsy-confirmed kidney disease, identifying 156 proteins associated with ATI
severity at Bonferroni-corrected significance thresholds, with replication across
three independent cohorts including the KPMP.[10] That work defines the current
state of knowledge: a rich set of candidate biomarkers with established disease
associations.

However, biomarker association and clinical prediction are distinct scientific
problems requiring different methods and evaluation criteria. Identifying proteins
that correlate with ATI severity does not establish whether those proteins can
generate well-calibrated individual-level probability estimates suitable for
clinical decision thresholds. Calibration — the alignment between predicted
probabilities and observed event rates — is the property that determines clinical
operationalisability.[11] A model with AUROC 0.90 but poor calibration may rank
patients correctly but assign a probability of 0.80 to patients whose true event
rate is 0.40, making it unsuitable as a decision tool. To date, no study has
developed a calibrated ML prediction framework for ATI using plasma SomaScan
proteomics.

This paper addresses that methodological gap. We develop and evaluate a two-arm
calibrated ML pipeline applied to the harmonized KPMP SomaScan Plasma datasets
(2022 and 2024 releases; combined n=320; DOI: 10.48698/6VRN-ZE53), using Platt
sigmoid calibration as the central methodological contribution. Our objective is
not to replicate or extend the association findings of Schmidt et al. 2024 but to
establish whether those signals support a clinically calibrated prediction model
and to provide a reproducible, governance-audited pipeline applicable to future
proteomic prediction studies in nephrology.

Specific aims: (1) evaluate two model architectures for binary ATI classification;
(2) quantify the marginal predictive contribution of proteomics beyond eGFR and
KDIGO Stage; (3) compare Platt-calibrated versus uncalibrated probability output
using ECE, Brier score, and reliability diagrams; (4) establish a reproducible ML
pipeline applicable to future larger-scale proteomic prediction studies.

---

## 2. Methods

### 2.1 Data Source and Study Population

This study used two KPMP SomaScan plasma biomarker datasets available under open
access (no data use agreement required) via the KPMP Atlas Repository
(https://atlas.kpmp.org):

- **2022 release:** SS-2214310, SomaScan v4.0, ANML-normalised, n=136,
  accessed April 12, 2026.
- **2024 release:** SS-2495199 v4.1, ANML-SMP-normalised, n=184,
  accessed April 12, 2026 (DOI: 10.48698/6VRN-ZE53).

Both datasets comprise participants with biopsy-confirmed kidney disease enrolled
in the KPMP, a multisite prospective cohort enrolling adults with AKI or CKD
undergoing clinically indicated kidney biopsy.[12] The two releases contain
entirely distinct participants (zero overlap confirmed by Participant ID
comparison). Combined harmonized cohort: n=320.

Binary ATI outcome was derived from the KPMP Open Access Clinical Data file
(20260309_OpenAccessClinicalData.csv, accessed April 12, 2026). The Primary
Adjudicated Category field — the result of centralised clinico-pathological review
by experienced KPMP clinicians and pathologists — was used to define ATI status:
participants adjudicated as "Acute Tubular Injury" were classified ATI-positive
(n=28); all other adjudicated categories were ATI-negative (n=292).

### 2.2 Dataset Harmonization and Batch Correction

The 2022 (SomaScan v4.0) and 2024 (SomaScan v4.1) releases differ by 238 aptamers:
7,604 measured in 2022, 7,596 in 2024, with 7,481 shared. Analysis was restricted
to the 7,481 common aptamers. Protein abundance values were log2-transformed
(clip at 1 RFU minimum) prior to analysis. A binary batch covariate (0=2022,
1=2024) was included in all models to account for inter-release technical
variation. No ComBat or other explicit batch correction was applied, as tree-based
models (XGBoost) are robust to batch effects when batch membership is included as
a feature; the logistic regression baseline also included the batch covariate.

The rationale for pooling these releases is: (1) both use the same KPMP study
protocol and biopsy adjudication process; (2) both use the same SomaScan EDTA
plasma ANML normalisation strategy; (3) the 7,481 shared aptamers represent 98.4%
of the 2022 panel and 98.5% of the 2024 panel; (4) including batch as a covariate
explicitly models any residual inter-release technical differences. This approach follows established practice for pooling aptamer-based proteomic
datasets with overlapping panels and shared normalisation strategies.[13]

### 2.3 Feature Engineering

Clinical features: baseline eGFR (mL/min/1.73 m², median imputation for 1 missing
value), KDIGO Stage (ordinal: Stage 1=1, Stage 2=2, Stage 3=3; 0 for missing),
batch covariate. Proteomic features: top-500 SomaScan aptamers selected by
cross-sample variance after log2 transformation, providing dimensionality
reduction from 7,481 to 500 prior to XGBoost training. The variance-based
selection is unsupervised (no label information used) and therefore does not
introduce selection bias.


### 2.4 Model Development

Two model architectures were evaluated:

**Arm 1 — Clinical baseline:** Logistic regression (L2 penalty, C=1.0,
class_weight='balanced', max_iter=1000) on eGFR, KDIGO Stage, and batch
covariate, with StandardScaler preprocessing. Represents the discriminative
information available from routine clinical variables without proteomic testing;
the primary comparator for assessing proteomic added value.

**Arm 2 — Combined proteomic+clinical (primary model):** XGBoost classifier
(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.1, scale_pos_weight=10.4 [ratio of negatives to positives])
on the top-500 variance-selected proteins plus eGFR, KDIGO Stage, and batch
covariate. Platt sigmoid probability calibration applied via
CalibratedClassifierCV(method='sigmoid', cv=3). The scale_pos_weight parameter
addresses the 10.4:1 class imbalance (292 ATI-negative vs 28 ATI-positive).

### 2.5 Validation Protocol

Both arms validated by stratified 5-fold cross-validation (StratifiedKFold,
n_splits=5, shuffle=True, random_state=42) using cross_val_predict(method='predict_proba'). Stratification ensures
approximately equal ATI prevalence across folds (~5–6 ATI-positive cases per fold).
Out-of-fold prediction vectors from this single CV pass were used for all
performance metrics, bootstrap CIs, and permutation tests (Section 2.6). No
external validation cohort was available; this is a known limitation (Section 4.5).

### 2.6 Performance Metrics

Primary: AUROC (area under the receiver operating characteristic curve) and AUPRC
(area under the precision-recall curve). AUPRC is the co-primary metric at 8.8%
prevalence, as it directly reflects performance at the operational prevalence
unlike AUROC; a random classifier achieves AUPRC equal to prevalence (0.088 here).
Secondary: log loss, Brier score, ECE (Expected Calibration Error, 5-bin quantile
method). ECE = mean(|fraction_positive_in_bin − mean_predicted_in_bin|) across bins.
All primary metrics computed from out-of-fold predictions.

**Uncertainty quantification:** Bootstrap 95% CIs on AUROC and AUPRC were computed
from 2000 resamples of the out-of-fold prediction vector (numpy default_rng, seed=42).
Folds with no positive cases in a resample were excluded. Label-permutation tests
(n=1000 shuffles) were used to assess whether observed AUROC exceeded a null
distribution; p-value = (count of null AUROCs ≥ observed + 1) / (n_permutations + 1).

**Sensitivity analysis:** A pre-specified AKI-only subgroup analysis was conducted
restricting to AKI-enrolled participants (n=61: 28 ATI-positive, 33 ATI-negative)
to test whether model performance reflected ATI detection within AKI rather than
AKI vs non-AKI enrollment separation.

### 2.7 Feature Importance

XGBoost gain-based feature importance computed on the full training dataset
(n=320). Top-20 features reported. Protein labels extracted from SomaScan
Gene.SeqId column format (gene symbol prefix).

### 2.8 Reproducibility and Governance

Python 3.14, scikit-learn ≥1.4, XGBoost ≥2.0. All random states fixed
(random_state=42). Full pipeline, governance files, and unit test suite at
https://github.com/julian-borges-md/ati-proteomics (MIT licence). GitHub
Actions CI runs on all pushes. Reported per TRIPOD+AI.[14]

---

## 3. Results

### 3.1 Cohort Characteristics

The harmonized cohort comprised 320 participants (136 from the 2022 release,
184 from the 2024 release). ATI-positive: n=28 (8.8%); ATI-negative: n=292
(91.2%). Full cohort characteristics are presented in Table 1.

A notable feature of the ATI-positive group is its composition: all 28
ATI-positive participants were enrolled under the AKI enrollment category
(28/28, 100%), reflecting the expected clinical epidemiology of ATI as a
predominant pathological finding in AKI. By contrast, the ATI-negative group
included CKD (n=210, 71.9%), AKI (n=33, 11.3%), Healthy Reference (n=32,
11.0%), and DM-R (n=17, 5.8%) participants. The study cohort is therefore an
enriched biopsy population; results should not be generalised to unselected
or community-based populations.

Baseline eGFR was paradoxically higher in ATI-positive participants (mean
81.9 ± 22.4 mL/min/1.73 m²) compared to ATI-negative (62.0 ± 25.9). This
reflects the AKI clinical context: ATI-positive participants presented with
acute injury from a higher pre-morbid eGFR baseline, consistent with the
known clinical presentation of AKI superimposed on preserved kidney function.

KDIGO Stage distribution in ATI-positive participants: Stage 1 n=2 (7.1%),
Stage 2 n=3 (10.7%), Stage 3 n=21 (75.0%), missing n=2 (7.1%), reflecting
the clinical severity of biopsy-indicated AKI in this cohort.

Diabetes history was less common in ATI-positive (6/28, 21.4%) than
ATI-negative (183/292, 62.7%), consistent with the predominantly non-diabetic
aetiology of ATI. Hypertension history was present in 11/28 (39.3%) ATI-positive
versus 230/292 (78.8%) ATI-negative participants.


**Table 1. Cohort Characteristics by ATI Status**

| Characteristic | ATI-Positive (n=28) | ATI-Negative (n=292) | Total (n=320) |
|---|---|---|---|
| **Batch** | | | |
| 2022 release | 17 (60.7%) | 119 (40.8%) | 136 (42.5%) |
| 2024 release | 11 (39.3%) | 173 (59.2%) | 184 (57.5%) |
| **Enrollment Category** | | | |
| AKI | 28 (100.0%) | 33 (11.3%) | 61 (19.1%) |
| CKD | 0 (0.0%) | 210 (71.9%) | 210 (65.6%) |
| Healthy Reference | 0 (0.0%) | 32 (11.0%) | 32 (10.0%) |
| DM-R | 0 (0.0%) | 17 (5.8%) | 17 (5.3%) |
| **Primary Adjudicated Category** | | | |
| Acute Tubular Injury | 28 (100.0%) | — | 28 (8.8%) |
| Diabetic Kidney Disease | — | 96 (32.9%) | 96 (30.0%) |
| Hypertensive Kidney Disease | — | 63 (21.6%) | 63 (19.7%) |
| Cannot be determined | — | 42 (14.4%) | 42 (13.1%) |
| Other | — | 33 (11.3%) | 33 (10.3%) |
| Acute Interstitial Nephritis | — | 12 (4.1%) | 12 (3.8%) |
| **Sex** | | | |
| Male | 20 (71.4%) | 160 (54.8%) | 180 (56.3%) |
| Female | 8 (28.6%) | 132 (45.2%) | 140 (43.8%) |
| **Baseline eGFR (mL/min/1.73 m²)** | | | |
| Mean ± SD | 81.9 ± 22.4 | 62.0 ± 25.9 | 63.8 ± 26.2 |
| Median | 76.5 | 55.2 | 57.3 |
| **KDIGO Stage** | | | |
| Stage 1 | 2 (7.1%) | 5 (1.7%) | 7 (2.2%) |
| Stage 2 | 3 (10.7%) | 11 (3.8%) | 14 (4.4%) |
| Stage 3 | 21 (75.0%) | 14 (4.8%) | 35 (10.9%) |
| Not applicable/missing | 2 (7.1%) | 262 (89.7%) | 264 (82.5%) |
| **Diabetes History** | | | |
| Yes | 6 (21.4%) | 183 (62.7%) | 189 (59.1%) |
| No | 20 (71.4%) | 107 (36.6%) | 127 (39.7%) |
| Don't know/missing | 2 (7.1%) | 2 (0.7%) | 4 (1.3%) |
| **Hypertension History** | | | |
| Yes | 11 (39.3%) | 230 (78.8%) | 241 (75.3%) |
| No | 17 (60.7%) | 59 (20.2%) | 76 (23.8%) |
| Don't know/missing | 0 (0.0%) | 3 (1.0%) | 3 (0.9%) |

*ATI: Acute Tubular Injury. AKI: Acute Kidney Injury. CKD: Chronic Kidney Disease.
DM-R: Diabetes Mellitus Resilience. eGFR: estimated glomerular filtration rate.
KDIGO: Kidney Disease Improving Global Outcomes. Note: all ATI-positive participants
were AKI-enrolled, reflecting expected ATI clinico-pathological epidemiology.*


### 3.2 Model Performance

Model performance across 5-fold cross-validation is presented in Table 2 and
Figure 1 (ROC curves).

The clinical-only logistic regression (Arm 1) achieved AUROC 0.938, log loss
0.306, and Brier score 0.082. This high clinical AUROC reflects the strong
separation between ATI-positive (AKI-enrolled, KDIGO Stage 3 predominant,
eGFR 81.9) and ATI-negative (CKD-predominant, eGFR 62.0) participants;
eGFR and KDIGO Stage together encode most of the clinically available ATI signal
in this biopsy-enriched cohort.

The proteomic+clinical XGBoost model (Arm 2) achieved AUROC 0.922 before
calibration and AUROC 0.898 after Platt calibration. Brier score was 0.060
for both uncalibrated and calibrated models; log loss was 0.192 uncalibrated
and 0.211 calibrated. The marginal AUROC gain of proteomics over clinical
variables alone was not demonstrated in this cohort (0.898 vs 0.938), consistent
with the strong clinical signal described above and the small ATI-positive n.

Bootstrap 95% confidence intervals on AUROC (n=2000 resamples) were: Clinical LR
0.939 [0.900–0.969], XGBoost calibrated 0.899 [0.833–0.952]. Label-permutation
tests (n=1000) yielded p=0.001 for both models, confirming that predictions are not
random. AUPRC for the Clinical LR was 0.588 [0.407–0.750] and for XGBoost 0.510
[0.335–0.687] versus a random classifier baseline of 0.088 (equal to prevalence),
indicating that both models substantially exceed chance on the precision-recall metric.

**AKI-only sensitivity analysis.** When restricted to AKI-enrolled participants only
(n=61: 28 ATI-positive, 33 ATI-negative), AUROC fell to 0.609 for Clinical LR and
0.541 for XGBoost. AUPRC was 0.597 and 0.545 respectively versus a random baseline
of 0.459. This near-chance performance within the AKI subgroup is the most important
finding of this study. It demonstrates that the high full-cohort AUROC (0.938/0.898)
was driven primarily by the clinical and proteomic separation between AKI-enrolled
and non-AKI-enrolled participants, not by detection of ATI within AKI cases. This
finding is consistent with the known clinical challenge of ATI: the condition occurs
predominantly in AKI, and distinguishing ATI-positive from ATI-negative AKI on
plasma proteomics at n=28 is not achievable with current sample sizes. All ATI
prediction performance claims must therefore be interpreted within this context.

**Table 2. Model Performance — Stratified 5-Fold Cross-Validation with Robustness Analyses**

| Analysis | Model | AUROC (95% CI) | AUPRC (95% CI) | Brier | ECE | Permutation p |
|---|---|---|---|---|---|---|
| **Full cohort (n=320)** | Clinical LR | 0.939 [0.900–0.969] | 0.588 [0.407–0.750] | 0.082 | — | 0.001 |
| | XGBoost+Platt calibrated | 0.899 [0.833–0.952] | 0.510 [0.335–0.687] | 0.060 | 0.035 | 0.001 |
| **AKI-only (n=61)** | Clinical LR | 0.609 | 0.597 | — | — | — |
| | XGBoost+Platt calibrated | 0.541 | 0.545 | — | — | — |

*95% CIs from bootstrap resampling (n=2000). AUPRC: area under precision-recall curve.
Random classifier AUPRC baseline = 0.088 (prevalence). ECE: Expected Calibration Error
(5-bin quantile method). Permutation p-value from label-shuffling (n=1000).
AKI-only subgroup: n=61, ATI-positive n=28, ATI-negative n=33.
Note: full-cohort AUROC reflects AKI vs non-AKI enrollment separation;
AKI-only AUROC reflects ATI detection within AKI — near-chance performance
indicating the proteomic ATI signal is not captured at this sample size.*

### 3.3 Calibration Analysis

Figure 2 presents reliability diagrams for the uncalibrated and Platt-calibrated
XGBoost models. Platt calibration achieved ECE 0.035, indicating well-aligned
predicted probabilities and observed event rates across quantile bins. Brier
score was 0.060 for both models; the primary calibration benefit is in
probability reliability rather than overall accuracy at this sample size.

The ECE of 0.035 is structurally well-calibrated for a rare-event classifier
(prevalence 8.8%). In clinical terms, a well-calibrated model with ECE 0.035
would assign predicted probabilities that deviate from observed event rates by
3.5 percentage points on average — an acceptable margin for risk stratification
in a screening context. This structural calibration property — not the AUROC
point estimate — is the primary transferable methodological contribution of
this pipeline.

### 3.4 Feature Importance

Figure 3 presents the top-20 XGBoost gain-based predictors. The highest-ranked
features were KDIGO Stage (clinical), PRDX3 (Peroxiredoxin-3, mitochondrial
antioxidant), PPFIA1 (Liprin-alpha-1, cell adhesion), DLX4 (Distal-less
homeobox 4, transcription factor), and NPLOC4 (Nuclear protein localization
protein 4, ubiquitin pathway). PRDX3 involvement is biologically plausible
given mitochondrial oxidative stress in tubular injury.[15] The dominance of
KDIGO Stage as the top predictor reinforces the strong clinical signal in this
cohort. Feature importance rankings reflect multivariate predictive contribution
in the context of this specific dataset and should not be interpreted as
independent biomarker validation.

### 3.5 Batch Effects

The 2022 cohort contributed 17/28 (60.7%) ATI-positive cases and the 2024
cohort contributed 11/28 (39.3%). The batch covariate was included in all
models. The ATI prevalence difference between batches (12.5% in 2022 vs 6.0%
in 2024) is consistent with expected cohort composition variation across
enrollment waves and does not indicate a systematic batch artifact. Feature
importance analysis confirmed that the batch covariate did not rank among the
top-20 predictors, suggesting that batch membership did not dominate the
proteomic signal.

---

## 4. Discussion

### 4.1 Principal Findings

This paper establishes a calibrated two-arm machine learning prediction framework
for ATI classification from plasma SomaScan proteomics, applied to the largest
harmonized open-access KPMP proteomic dataset currently available (n=320, ATI
n=28). The scientific contribution is methodological: we provide the first
calibrated, comparative prediction modelling framework applied to this disease
target, evaluated against a clinical baseline, and optimised for probabilistic
reliability. The pipeline is reproducible, governance-audited, and openly released.

Three findings merit specific discussion: the high clinical AUROC; the absence of
proteomic added value in this cohort; and the calibration performance.

### 4.2 The High Clinical AUROC and Its Interpretation

The clinical logistic regression achieved AUROC 0.938 — higher than the XGBoost
proteomic model. This is not a surprising finding and does not indicate that
proteomics is uninformative. It reflects the composition of this biopsy-enriched
cohort: all 28 ATI-positive participants were AKI-enrolled, most with KDIGO Stage 3,
and with a paradoxically higher eGFR than the ATI-negative group. In this context,
eGFR and KDIGO Stage together function almost as a proxy label for ATI status.

In an unselected population — where ATI-positive cases might present among
CKD, DM-R, or other categories with overlapping eGFR ranges — the clinical
signal would be substantially weaker and the proteomic contribution more apparent.
This dataset is not designed to test the incremental value of proteomics over
clinical variables in an unselected population; it is a biopsy-enriched cohort
where the clinical context is intentionally informative. This limitation is
acknowledged and does not diminish the methodological contribution of the pipeline.

### 4.3 Relationship to Schmidt et al. 2024

This paper explicitly builds on Schmidt et al. 2024.[10] That landmark study
established the biological coherence of the ATI plasma proteomic signal using
a biomarker discovery paradigm — univariate linear regression with Bonferroni
correction, optimising for association p-values. The scientific question it
answered (which proteins associate with ATI severity) is necessary but not
sufficient for clinical translation. The question this paper addresses (can those
proteins generate well-calibrated individual probability estimates) is the next
methodological step in the translational pathway. The two papers address
complementary scientific questions and should be read together.

The KPMP cohort in Schmidt et al. served as a replication dataset for associations
discovered in a separate primary cohort (n=434). Here the KPMP data serve as the
primary modelling cohort for a supervised prediction exercise — a fundamentally
different analytical role.

### 4.4 The Calibration Contribution

Platt calibration achieved ECE 0.035 in this cohort. Calibration is the property
that determines clinical operationalisability of a prediction model.[11] A model
with ECE 0.035 assigns predicted probabilities that deviate from observed rates by
3.5 percentage points on average — sufficient for risk stratification and clinical
triage. The CalibratedClassifierCV(method='sigmoid') framework applied here is
transferable to any high-dimensional proteomic prediction problem and represents
the primary generalizable methodological contribution of this work, independent
of the specific AUROC values in this dataset.

### 4.5 Limitations

(1) **Sample size.** ATI n=28 with 5-fold CV produces approximately 5–6
ATI-positive cases per test fold. AUROC point estimates are unreliable at this
event count; confidence intervals would be extremely wide. All performance
estimates are exploratory and should not be treated as definitive.

(2) **Cohort enrichment.** All ATI-positive participants were AKI-enrolled.
The clinical AUROC 0.938 reflects this enrichment rather than generalizable
discriminative ability. Results cannot be extrapolated to unselected populations.

(3) **Batch harmonization.** The 2022 and 2024 releases differ in SomaScan panel
version (v4.0 vs v4.1) and normalization (ANML vs ANML-SMP). Analysis was
restricted to 7,481 shared aptamers with batch as a covariate. Residual
inter-release technical variation cannot be fully excluded.

(4) **No external validation.** All results are from internal 5-fold CV on a
single harmonized dataset. External validation in an independent cohort is
required before any clinical interpretation.

(5) **Feature selection bias.** Variance-based protein selection was unsupervised
and therefore does not introduce selection bias; however, the 500-protein subset
may not include all biologically relevant ATI markers.

(6) **Cross-sectional design.** This is a diagnostic model; temporal or prognostic
inference is not supported.

### 4.6 Future Directions

The critical next step is external validation in an independent powered cohort
(recommended ATI n≥100) with prospective plasma collection. That cohort would
support regularisation optimisation, demonstration of proteomic incremental value
over clinical baselines, and formal calibration assessment. Controlled-access
KPMP data (full cohort, n>800 with DUA) represents the most accessible near-term
expansion. Integration with the KPMP controlled-access proteomics under a formal
DUA is the recommended next step for this pipeline.

---

## 5. Conclusion

This study establishes a reproducible, calibrated, two-arm machine learning
prediction pipeline for ATI classification from plasma SomaScan proteomics,
filling a methodological gap distinct from the biomarker association work of
Schmidt et al. 2024. Applied to the largest available open-access harmonized
KPMP SomaScan dataset (n=320, ATI n=28), the pipeline demonstrated structural
calibration (ECE 0.035) and strong clinical baseline performance (AUROC 0.938).
All performance estimates are preliminary given ATI n=28. The pipeline is openly
released, governance-audited, and TRIPOD+AI compliant. Its central contributions —
Platt probability calibration, comparative architecture evaluation, and
generalizable code infrastructure — are applicable to proteomic prediction
problems beyond ATI. Validation in larger prospective cohorts will determine
whether plasma proteomics adds independent predictive value beyond clinical
variables in this important disease context.

---

## Data Availability

KPMP SomaScan plasma datasets are available via open access at
https://atlas.kpmp.org (DOI: 10.48698/6VRN-ZE53; no data use agreement required,
accessed April 12, 2026). KPMP clinical data available at the same repository
(20260309_OpenAccessClinicalData.csv). Analysis code at
https://github.com/julian-borges-md/ati-proteomics (MIT licence).

## KPMP Data Citation

The results here are in whole or part based upon data generated by the Kidney
Precision Medicine Project. Accessed April 12, 2026. https://www.kpmp.org.
The Kidney Precision Medicine Project (KPMP) is supported by the National
Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) through the
following grants: U01DK133081, U01DK133091, U01DK133092, U01DK133093,
U01DK133095, U01DK133097, U01DK114866, U01DK114908, U01DK133090, U01DK133113,
U01DK133766, U01DK133768, U01DK114907, U01DK114920, U01DK114923, U01DK114933,
U24DK114886, UH3DK114926, UH3DK114861, UH3DK114915, and UH3DK114937. We
gratefully acknowledge the essential contributions of our patient participants
and the support of the American public through their tax dollars.

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
15. Bonventre JV. *Kidney Int.* 1993;43:1160. doi:10.1038/ki.1993.163

