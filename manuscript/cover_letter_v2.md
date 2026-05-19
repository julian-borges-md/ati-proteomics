# Cover Letter — ATI_proteomics_draft_v4
# Target: PLOS ONE or Frontiers in Nephrology
# Author: Julian Borges, MD | jyborges@bu.edu

---

Dear Editors,

We submit for your consideration the manuscript titled:

**"A Calibrated Machine Learning Framework for Plasma Proteomic Prediction of
Acute Tubular Injury: Methodological Development and Proof-of-Concept in the
Kidney Precision Medicine Project"**

**Study overview**

We developed and evaluated a two-arm calibrated machine learning prediction
pipeline for binary ATI classification applied to the largest harmonized
open-access KPMP SomaScan plasma dataset currently available: the 2022 (n=136)
and 2024 (n=184) releases combined into a harmonized cohort of n=320, ATI n=28
(8.8% prevalence). Data were accessed April 12, 2026 via open access at
https://atlas.kpmp.org (DOI: 10.48698/6VRN-ZE53); no data use agreement was
required. The study was conducted and reported per TRIPOD+AI guidelines. Analysis
code is openly released at https://github.com/julian-borges-md/ati-proteomics
(MIT licence). No new patient data were collected.

**Mandatory prior work disclosure**

We disclose proactively that the KPMP SomaScan dataset used here was previously
analysed by Schmidt IM et al. (*Nat Commun.* 2024;15:7368) as a replication
cohort for ATI biomarker associations identified in a separate primary cohort
(n=434). The Schmidt et al. study and the present manuscript address entirely
distinct scientific questions. Schmidt et al. used univariate linear regression
to identify protein-ATI associations. The present study uses supervised
multivariate classification with Platt sigmoid probability calibration to evaluate
whether those signals support clinically calibrated individual-level probability
estimates. No data, code, or unpublished results from the Schmidt et al. group
were used. Full differentiation is provided in the manuscript Discussion.

**Principal findings**

The clinical logistic regression (Arm 1: eGFR, KDIGO Stage) achieved AUROC 0.939
[0.900–0.969]. The XGBoost proteomic+clinical model with Platt calibration (Arm 2)
achieved AUROC 0.899 [0.833–0.952], AUPRC 0.510 [0.335–0.687], Brier 0.060,
ECE 0.035. Label-permutation tests confirmed non-random predictions (p=0.001
both models). A pre-specified AKI-only sensitivity analysis (n=61) revealed
near-chance performance (AUROC 0.609 clinical, 0.541 XGBoost), demonstrating
that full-cohort performance reflects AKI vs non-AKI enrollment separation rather
than ATI detection within AKI — an honest and important finding that is stated
prominently in Results and Discussion.

**Scientific contribution**

The manuscript's contribution is methodological. We provide the first calibrated,
comparative prediction modelling framework for ATI from plasma SomaScan proteomics,
with structural calibration (ECE 0.035), complete robustness analysis (bootstrap
CIs, permutation tests, AKI-only sensitivity analysis, AUPRC), and fully
reproducible open-source code. All results are presented as preliminary given
ATI n=28; no clinical claims are made. The pipeline infrastructure is
transferable to any high-dimensional proteomic prediction problem.

We believe this manuscript is appropriate for publication as a methods and
proof-of-concept contribution. We are happy to provide any additional information.

Sincerely,

Julian Borges, MD
Frontier Translational Research Lab
Department of Computer Science, Boston University
jyborges@bu.edu | ORCID: 0009-0001-9929-3135
