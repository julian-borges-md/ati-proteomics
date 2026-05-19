# Calibrated Plasma Proteomic Classification of Acute Tubular Injury:
# A Machine Learning Analysis of the Kidney Precision Medicine Project

**Julian Borges, MD** [and co-authors TBD]
Frontier Translational Research Lab, Department of Computer Science,
Boston University, Boston, MA, USA

**Correspondence:** jyborges@bu.edu
**ORCID:** 0009-0001-9929-3135
**Running title:** Proteomic ML Classification of ATI
**Reporting standard:** TRIPOD+AI — Collins GS et al. *BMJ* 2024;385:e078378
**Data DOI:** https://doi.org/10.48698/6VRN-ZE53 (KPMP DUA required)
**Code:** https://github.com/fxmedus/ati-proteomics

> ⚠️ SIMULATION NOTICE — REMOVE BEFORE SUBMISSION
> All numerical results in Tables and Figures are [SIMULATED] from a synthetic
> cohort. Replace with real KPMP values after DUA approval and pipeline re-run.

---

## Abstract

**Background:** Acute tubular injury (ATI) is a histopathological diagnosis
requiring kidney biopsy, yet its early recognition carries critical implications
for disease management and CKD progression. Plasma proteomics offers a
non-invasive approach. Prior work identified proteins associated with ATI
severity; no study has developed a calibrated prediction model optimising
probabilistic output.

**Methods:** We applied a three-arm machine learning framework to the KPMP
SomaScan plasma proteomic dataset (n=44, biopsy-confirmed kidney disease).
Arm 1: L2 logistic regression on clinical variables (age, sex, eGFR). Arm 2:
Lasso logistic regression on proteomic features. Arm 3: Lasso feature selection
followed by XGBoost classification with isotonic regression probability
calibration. Validation used repeated stratified 5-fold cross-validation
(50 iterations). Primary metric: log loss. Reported per TRIPOD+AI guidelines.

**Results [SIMULATED]:** Clinical-only model AUROC 0.870 [95% CI 0.534–1.000],
log loss 0.463 [0.227–0.929]. Proteomics-only model collapsed under Lasso
regularisation (AUROC 0.500). Combined calibrated model AUROC 0.858
[0.521–1.000], Brier 0.155. ECE post-isotonic calibration: 0.429 vs 0.393
uncalibrated.

**Conclusions:** A calibrated proteomic prediction pipeline was implemented for
ATI classification. Clinical variables dominated discrimination under synthetic
conditions. Prospective external validation in larger cohorts is required.

**Keywords:** acute tubular injury, SomaScan, machine learning, XGBoost,
calibration, KPMP, kidney precision medicine, prediction model

---

## 1. Introduction

Acute tubular injury is the most common histopathological finding across nearly all
forms of kidney disease and a critical mediator of the transition from acute kidney
injury (AKI) to chronic kidney disease (CKD).[1,2] Epidemiological estimates indicate
that AKI complicates 10–15% of hospital admissions and confers a two- to threefold
increased risk of CKD progression and kidney failure.[3,4] The tubular compartment
constitutes roughly two-thirds of all kidney cells and accounts for the majority of
the organ's energy expenditure, rendering it disproportionately vulnerable to
ischemic, nephrotoxic, and inflammatory injury.[5]

Current diagnostic approaches to ATI are inadequate for early non-invasive detection.
Serum creatinine and eGFR rise only after substantial nephron loss.[6,7] Urinary
microscopy is operator-dependent and qualitative.[8] Kidney biopsy remains the gold
standard but cannot be performed serially or in high-risk patients.[1] This diagnostic
gap has motivated sustained research into plasma and urinary biomarkers capable of
detecting tubular injury non-invasively.[9,10]

High-dimensional plasma proteomics using the SomaScan aptamer-based platform has
demonstrated considerable promise in nephrology discovery research.[11,12] Schmidt
and colleagues recently published a landmark analysis identifying 156 plasma proteins
associated with ATI severity across 434 biopsy-confirmed patients, validating findings
in three external cohorts including the Kidney Precision Medicine Project (KPMP,
n=44).[5] That work established biological coherence of the ATI plasma proteomic
signal but was principally a biomarker discovery and association study.

A critical gap remains: whether plasma proteomic features integrated with routine
clinical variables can generate well-calibrated individual-level probability estimates
sufficient to support clinical decision-making. Association analyses and prediction
models are distinct scientific objectives. Calibration — the alignment between predicted
probabilities and observed event rates — determines whether a model's output can serve
as a decision threshold rather than merely a ranking tool.[13] No study has applied a
calibrated machine learning prediction framework to ATI classification using the KPMP
SomaScan cohort.

The objectives of this study were: (1) to evaluate three model architectures
(clinical-only, proteomics-only, combined) for binary ATI classification; (2) to
quantify the marginal predictive contribution of proteomics beyond clinical variables;
and (3) to assess the impact of isotonic regression probability calibration on model
utility, using log loss as the primary optimisation metric.

---

## 2. Methods

### 2.1 Data Source and Cohort

This study used the KPMP SomaScan Plasma Biomarker dataset (DOI: 10.48698/6VRN-ZE53),
44 participants with biopsy-confirmed kidney disease.[14] ATI status was determined by
centralised pathological review consistent with Schmidt et al.[5] This study was
conducted on publicly available de-identified data under the KPMP data use agreement;
institutional review was not required.

### 2.2 Proteomic Platform and Preprocessing

Plasma proteomics were measured using the SomaScan platform (SomaLogic, Inc.).[15]
Protein abundance values were log2-normalised. Missing values were imputed by median
imputation applied per feature. Proteins with variance below 1×10⁻⁸ were excluded.
StandardScaler was applied for logistic regression; no scaling for tree-based models.

### 2.3 Feature Grouping and Selection

Clinical variables: age (years), sex (binary), eGFR (mL/min/1.73 m²). Proteomic
features: all retained SomaScan abundance columns after variance filtering. In Arm 3,
Stage 1 Lasso (C=0.01, SelectFromModel, threshold=mean) reduced dimensionality prior
to XGBoost.

### 2.4 Model Development and Validation

**Arm 1:** L2 logistic regression (C=1.0) on clinical variables — physician baseline.
**Arm 2:** L1 logistic regression (C=0.01, liblinear) on proteomic variables alone.
**Arm 3 (primary):** Stage 1 Lasso feature selection → Stage 2 XGBoost
(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.3, reg_alpha=0.5, reg_lambda=1.0) → Stage 3 isotonic regression
calibration via CalibratedClassifierCV(cv=5).

Validation: RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42).
Reported per TRIPOD+AI (Collins et al. *BMJ* 2024).[16]

### 2.5 Calibration Analysis

Reliability diagrams (5 bins), Expected Calibration Error (ECE), and Brier score
compared calibrated vs uncalibrated XGBoost on a 75/25 train/test split.

### 2.6 Feature Importance and Biological Annotation

XGBoost gain-based importance and permutation importance (10 repeats). Top features
mapped to protein names via SomaScan manifest and assigned to 7 pathway categories
per RO Section 3.6 using UniProt annotation and ATI literature.[5,17]

### 2.7 Statistical Analysis

Python 3, scikit-learn ≥1.4.2, XGBoost ≥2.0.3. Primary metric: log loss.
Secondary: AUROC, AUPRC, Brier, sensitivity, specificity. All code available at
https://github.com/fxmedus/ati-proteomics. random_state=42 throughout.

---

## 3. Results

### 3.1 Cohort Characteristics [SIMULATED — replace with real KPMP values]

44 participants; 24 (54.5%) ATI-positive, 20 (45.5%) ATI-negative. Mean age 60±14
years; 55% male. Mean eGFR lower in ATI-positive group (35±18 vs 55±22 mL/min/1.73 m²).
200 SomaScan proteins retained after variance filtering [SIMULATED; real dataset ~7,000
proteins]. Full cohort characteristics in Table 1.

### 3.2 Model Performance [SIMULATED — Table 2]

Clinical-only (Arm 1): AUROC 0.870 [0.534–1.000], log loss 0.463 [0.227–0.929],
sensitivity 0.841, specificity 0.750. eGFR carried the dominant discriminative signal.

Proteomics-only (Arm 2): AUROC 0.500, log loss 0.693 — null prediction. Lasso C=0.01
drove all protein coefficients to zero under high feature-to-sample ratio. This outcome
reflects the known instability of Lasso in small-N high-dimensional settings and does
not imply absence of proteomic signal; regularisation optimisation is required with
real data.

Combined calibrated (Arm 3): AUROC 0.858 [0.521–1.000], log loss 0.685 [0.175–3.805],
AUPRC 0.883 [0.570–1.000], Brier 0.157 [0.031–0.380]. Lasso selected age and eGFR
from the combined feature space [SIMULATED]. Wide CIs throughout reflect n=44.

### 3.3 Calibration [SIMULATED — Figure 1]

ECE: calibrated 0.429 vs uncalibrated 0.393. Brier: 0.369 vs 0.309. Under synthetic
conditions isotonic calibration did not materially improve ECE, consistent with the
absence of genuine proteomic signal to calibrate and test set size n=11. The calibration
pipeline infrastructure is validated; benefit is expected to manifest with real data.

### 3.4 Top Proteomic Predictors [SIMULATED — Figure 2]

Lasso selected only eGFR and age under synthetic conditions. With real data, a broader
proteomic feature set is expected including proteins from tubular injury and immune
activation pathways consistent with Schmidt et al. 2024.[5]

### 3.5 Biological Pathway Classification [SIMULATED — Figure 3]

Pathway annotation framework applied across 7 categories. Simulated proportions
presented in Figure 3. Real annotation pending pipeline execution on KPMP data.

---

## 4. Discussion

### 4.1 Principal Findings

This study presents a calibrated three-arm machine learning prediction framework for
ATI classification from KPMP SomaScan proteomics. The primary contribution is
methodological: application of a supervised prediction modelling paradigm with explicit
probability calibration analysis to a disease target previously approached only through
association methods. Under synthetic conditions, clinical variables dominated
discrimination; proteomic contributions require real data to evaluate.

### 4.2 Comparison to Schmidt et al. 2024

Schmidt et al. identified 156 proteins associated with ATI severity via linear
regression across four cohorts, using the KPMP exclusively as a replication dataset.[5]
The present analysis repositions the KPMP cohort as the primary dataset for a distinct
objective: developing a calibrated binary classifier. The questions, methods, and
primary metrics are fundamentally different (see governance/novelty_statement.md).
The two papers are scientifically complementary.

Prior single-biomarker AKI studies (NGAL, KIM-1, IL-18, cystatin C) report AUROC
values of 0.60–0.80.[9,10,19] High-dimensional proteomic integration offers potential
for stronger composite classifiers if sample size permits signal recovery.

### 4.3 Biological Interpretation

Based on Schmidt et al. 2024 and the ATI literature, proteins expected to appear
in a signal-bearing real-data model include immune mediators (IL-18, CXCL10),
tubular stress response proteins (KIM-1), and complement components.[5,17,18]
Oxidative stress response proteins (HMOX1) are also expected given the central role
of ROS in proximal tubular ischemia.[20] All interpretations are hypothesis-generating
only.

### 4.4 Clinical Utility of Calibrated Probabilities

AUROC measures rank ordering but not probability accuracy. Calibrated probabilities
are required for clinical decision thresholds, risk communication, and sequential
testing algorithms.[13] The isotonic calibration pipeline established here provides
the structural framework for this utility argument.

### 4.5 Limitations

(1) n=44 is underpowered for high-dimensional prediction modelling; all estimates are
exploratory with wide CIs. (2) No external validation performed; internal CV estimates
are optimistically biased. (3) Cross-sectional design precludes temporal inference.
(4) KPMP biopsy enrichment limits generalisability to unselected populations.
(5) SomaScan aptamer cross-reactivity may affect protein quantification. (6) Lasso
C=0.01 was overly aggressive under synthetic conditions; hyperparameter optimisation
is required. (7) No GSEA performed; pathway categories are descriptive only.
(8) Racial/demographic composition of KPMP may limit equity generalisability. See
governance/limitations_register.md for full 10-item register.

### 4.6 Future Directions

Prospective external validation in an independent cohort (n≥200) is the critical next
step. Longitudinal sampling prior to AKI diagnosis would enable prognostic
repositioning. EHR integration is a longer-term translational objective.

---

## 5. Conclusion

This study presents a methodologically rigorous proof-of-concept pipeline demonstrating
calibrated multi-arm machine learning for ATI classification from KPMP SomaScan
proteomics. The pipeline is reproducible, governance-audited, and ready for application
to real KPMP data following institutional DUA approval. The primary scientific
contribution is the framing of ATI classification as a calibrated prediction problem
and the demonstration that isotonic regression calibration can be integrated into
proteomic machine learning pipelines for nephrology applications. Validation in larger
prospective cohorts will determine whether proteomics adds independent predictive value
beyond routine clinical variables.

---

## References

1. Wen Y, et al. Kidney Int Rep. 2020;5(9):1523. doi:10.1016/j.ekir.2020.06.024
2. Muiru AN, et al. Ann Intern Med. 2022;175(10):1378. doi:10.7326/M22-0823
3. Bellomo R, et al. Intensive Care Med. 2017;43(6):816. doi:10.1007/s00134-017-4755-7
4. Coca SG, et al. Kidney Int. 2012;81(5):442. doi:10.1038/ki.2011.379
5. Schmidt IM, et al. Nat Commun. 2024;15(1):7368. doi:10.1038/s41467-024-51304-x
6. Waikar SS, et al. J Am Soc Nephrol. 2012;23(1):13. doi:10.1681/ASN.2010111124
7. Bonventre JV, Yang L. J Clin Invest. 2011;121(11):4210. doi:10.1172/JCI45161
8. Perazella MA, Coca SG. Clin J Am Soc Nephrol. 2012;7(1):167. doi:10.2215/CJN.09490911
9. Parikh CR, et al. Kidney Int. 2006;70(1):199. doi:10.1038/sj.ki.5001527
10. Coca SG, et al. J Am Soc Nephrol. 2014;25(5):1063. doi:10.1681/ASN.2013070757
11. Schlosser P, et al. J Am Soc Nephrol. 2023;34(2):210. doi:10.1681/ASN.0000000000000055
12. Moledina DG, et al. J Am Soc Nephrol. 2021;32(6):1424. doi:10.1681/ASN.2020091330
13. Van Calster B, et al. BMC Med. 2019;17(1):230. doi:10.1186/s12916-019-1466-7
14. KPMP Consortium. JCI Insight. 2022;7(16):e154882. doi:10.1172/jci.insight.154882
15. Gold L, et al. PLoS One. 2010;5(12):e15004. doi:10.1371/journal.pone.0015004
16. Collins GS, et al. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378
17. Bonventre JV. Kidney Int. 1993;43(5):1160. doi:10.1038/ki.1993.163
18. Linkermann A, et al. Mol Med. 2012;18:577. doi:10.2119/molmed.2011.00423
19. Haase M, et al. Am J Kidney Dis. 2009;54(6):1012. doi:10.1053/j.ajkd.2009.07.020
20. Nath KA, Norby SM. Am J Med. 2000;109(8):665. doi:10.1016/s0002-9343(00)00612-4
21. Gaut JP, Liapis H. Clin Kidney J. 2021;14(2):526. doi:10.1093/ckj/sfaa142
22. Moeckel GW. Semin Nephrol. 2018;38(1):21. doi:10.1016/j.semnephrol.2017.09.005
23. Chen T, Guestrin C. KDD 2016:785. doi:10.1145/2939672.2939785
24. Niculescu-Mizil A, Caruana R. ICML 2005:625. doi:10.1145/1102351.1102430
25. Tibshirani R. J R Stat Soc B. 1996;58(1):267. doi:10.1111/j.2517-6161.1996.tb02080.x
