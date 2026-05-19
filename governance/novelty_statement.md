# Novelty Statement — RO-2026-001
**Gate:** G2 Pre-Writing Gate (mandatory before Introduction draft)
**Date:** 2026-04-12 | **Status:** CLEARED

---

## Anchor Publication

Schmidt IM, Surapaneni AL, Zhao R, et al. Plasma proteomics of acute tubular injury.
*Nat Commun.* 2024;15(1):7368. DOI: 10.1038/s41467-024-51304-x

---

## What Schmidt et al. 2024 Did

Schmidt and colleagues conducted a large-scale biomarker discovery study using SomaScan on
434 patients with biopsy-confirmed kidney disease from the Boston Kidney Biopsy Cohort (BKBC).
Their primary objective was to identify plasma proteins statistically associated with ATI severity
via linear regression, adjusting for age, sex, race, and eGFR. They identified 156 proteins at
Bonferroni-corrected thresholds and validated associations in three external cohorts: KPMP (n=44),
ARIC (n=4610), and CHROME (n=268). The KPMP cohort was used exclusively as a replication
dataset. Schmidt et al. did not develop a clinical prediction model, compare model architectures,
apply machine learning classifiers with cross-validated performance evaluation, optimize for
probabilistic output, apply probability calibration, or quantify the incremental predictive value
of proteomics beyond clinical variables.

---

## Five Explicit Differentiators

**1. Prediction versus association.**
Schmidt et al. used univariate linear regression to identify proteins associated with ATI severity.
This paper develops a multivariate supervised classifier that generates individual-level probability
estimates. Association studies identify which proteins correlate with disease at the population level;
prediction models determine whether those proteins can classify individual patients — a distinct
scientific objective requiring distinct methods and evaluation criteria.

**2. Comparative architecture evaluation.**
This paper evaluates three model architectures (clinical-only, proteomics-only, combined) within
a single repeated cross-validation framework, quantifying the marginal predictive contribution of
proteomics beyond age, sex, and eGFR alone. Schmidt et al. provided no such comparison.
Demonstrating incremental value of proteomics over a clinical baseline is a prerequisite for arguing
that proteomic testing would change clinical management.

**3. Probability calibration as primary methodological contribution.**
Calibration is entirely absent from Schmidt et al. This paper treats calibration as the central
methodological differentiator, comparing uncalibrated versus isotonic regression-calibrated XGBoost
using reliability diagrams, Expected Calibration Error, and Brier score. Calibrated probabilities
are required for clinical decision thresholds; uncalibrated classifiers may discriminate well but
produce unreliable individual risk estimates. This is the paper's primary scientific contribution
beyond Schmidt et al. 2024.

**4. Log-loss optimization framing.**
Schmidt et al. optimized for statistical discovery (Bonferroni p-values). This paper optimizes
for probabilistic sharpness (log loss), which directly penalizes confident mispredictions and is
more aligned with clinical utility in a risk stratification context.

**5. Supervised feature importance in a predictive context.**
Schmidt et al. ranked proteins by regression coefficient magnitude in a univariate framework. This
paper ranks proteins by XGBoost gain-based importance and permutation importance within a
multivariate predictive model, reflecting each protein's contribution to classification accuracy
rather than marginal association with ATI severity. These rankings are expected to differ and
represent complementary biological intelligence.

---

## Overlap Acknowledged

Both papers use SomaScan plasma proteomics from the KPMP cohort (n=44) and share the ATI binary
outcome. Overlapping protein candidates in the results are expected and appropriate — they validate
biological coherence of the predictive model. The papers are differentiated by scientific objective,
methodology, analytic framing, and clinical utility argument.

**Gate cleared. Introduction draft approved to proceed.**
