# Cover Letter — ATI_proteomics_draft_v1
# Target: Clinical Kidney Journal (Oxford University Press)
# Author: Julian Borges, MD | jyborges@bu.edu

---

Dear Editors of Clinical Kidney Journal,

We submit for your consideration the manuscript titled:

**"A Calibrated Machine Learning Framework for Plasma Proteomic Prediction of
Acute Tubular Injury: Methodological Development and Proof-of-Concept in the
Kidney Precision Medicine Project"**

**Mandatory disclosure of related prior work**

We are obligated to disclose, and do so proactively, that the dataset used in
this analysis — the KPMP SomaScan Plasma Biomarker dataset (DOI:
10.48698/6VRN-ZE53, n=44) — was previously analyzed and published by Schmidt IM
et al. (*Nat Commun.* 2024;15:7368). That paper identified 156 plasma proteins
associated with ATI severity via linear regression in a primary biopsy cohort
(n=434) and used the KPMP cohort exclusively as a replication dataset. The
Schmidt et al. paper and the present manuscript address entirely distinct
scientific questions, as detailed below.

**Scientific differentiation**

Schmidt et al. (2024) conducted a biomarker discovery and association study. The
scientific question was: which plasma proteins are statistically associated with
ATI severity? The analytical approach was univariate linear regression with
Bonferroni correction. No prediction model was developed, no comparative model
architectures were evaluated, and probability calibration was not performed.

The present manuscript addresses a different scientific question: can a calibrated
plasma proteomic classifier, integrating clinical variables and proteomic features
within a supervised machine learning framework, generate individual-level
probability estimates suitable for clinical risk stratification? Our primary
outcome metric is log loss — a direct measure of probabilistic sharpness — rather
than association p-values. Our central methodological contribution is the
demonstration and evaluation of isotonic regression probability calibration in a
proteomic prediction context: a contribution that is entirely absent from Schmidt
et al. and, to our knowledge, from the ATI biomarker literature broadly.

We explicitly frame Schmidt et al. as the scientific foundation on which this
work builds. Their discovery of ATI-associated proteins motivates the hypothesis
that these proteins, integrated in a multivariate calibrated classifier, could
support non-invasive clinical decision-making. Testing that hypothesis requires
the prediction modelling methodology we describe.

**Independent conduct**

This work was conducted independently using publicly available data obtained
under the standard KPMP Data Use Agreement. No data, analysis code, or
unpublished results from the Schmidt et al. study group were used in this work.
The KPMP dataset is publicly available to any credentialed researcher.

**Significance**

Calibrated prediction models — not biomarker associations — are what clinicians
require to operationalise proteomic testing in practice. A model that predicts
ATI probability of 0.75 is only clinically actionable if that probability
genuinely reflects a 75% observed event rate. The calibration infrastructure
this paper establishes, and the comparative evaluation of model architectures
it provides, fill a methodological gap that association studies cannot address.

We believe this manuscript is appropriate for Clinical Kidney Journal given its
nephrology focus and its interest in translational computational methods. We are
happy to provide any additional information the editors require.

Sincerely,

Julian Borges, MD
Frontier Translational Research Lab
Department of Computer Science, Boston University
jyborges@bu.edu | ORCID: 0009-0001-9929-3135
