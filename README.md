# Calibrated Plasma Proteomic Classification of Acute Tubular Injury

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TRIPOD+AI](https://img.shields.io/badge/Reporting-TRIPOD%2BAI-green)](https://doi.org/10.1136/bmj-2023-078378)
[![DOI Dataset](https://img.shields.io/badge/Dataset-10.48698%2F6VRN--ZE53-orange)](https://doi.org/10.48698/6VRN-ZE53)
[![Anchor Paper](https://img.shields.io/badge/Anchor-Schmidt%20et%20al.%202024-lightblue)](https://doi.org/10.1038/s41467-024-51304-x)
[![Research Order](https://img.shields.io/badge/Order-RO--2026--001-purple)](governance/decision_log.md)
[![Status](https://img.shields.io/badge/Status-Awaiting%20KPMP%20DUA-red)](https://www.kpmp.org/available-data)

> **Frontier Translational Research Lab**  
> Principal Investigator: Julian Borges, MD (`jyborges@bu.edu`)  
> Research Order: RO-2026-001 | Date: 2026-04-12

---

## Overview

This repository contains the complete, reproducible machine learning pipeline for the manuscript:

> **"Calibrated Plasma Proteomic Classification of Acute Tubular Injury: A Machine Learning Analysis of the Kidney Precision Medicine Project"**  
> Julian Borges, MD [and co-authors TBD]  
> Target journal: *Clinical Kidney Journal* (Oxford University Press)

### Scientific Objective

Can a calibrated, high-dimensional plasma proteomic classifier trained on publicly available
SomaScan data predict acute tubular injury (ATI) with clinically meaningful discrimination,
and does proteomic signal add predictive value beyond routine clinical variables (age, sex, eGFR)?

### Key Differentiator from Schmidt et al. 2024

Schmidt et al. (*Nat Commun* 2024;15:7368) identified 156 proteins *associated* with ATI severity
via linear regression. This pipeline develops a *calibrated binary prediction model*, evaluates
three comparative architectures, and optimizes for log loss (probabilistic sharpness) rather than
association p-values. See [`governance/novelty_statement.md`](governance/novelty_statement.md) for
the full pre-writing novelty gate document.

---

## ⚠️ Data Access — DUA Required

The KPMP SomaScan dataset requires institutional data use agreement (DUA) before download.

1. Navigate to <https://www.kpmp.org/available-data>
2. Apply for access under the dataset DOI: `10.48698/6VRN-ZE53`
3. Upon approval, place the file at: `data/raw/kpmp_somascan.csv`
4. All scripts auto-detect the file; if absent, a synthetic cohort is generated for pipeline demonstration

**The raw data file is excluded from this repository by `.gitignore` and must never be committed.**

---

## Repository Structure

```
ati-proteomics/
├── README.md                        # This file
├── requirements.txt                 # Pinned dependencies
├── pyproject.toml                   # Package metadata
├── LICENSE                          # MIT
│
├── src/                             # All analysis code
│   ├── train.py                     # Full ML pipeline (preprocessing → CV → calibration)
│   ├── figures.py                   # Publication-quality figure generation (300 DPI)
│   └── annotate.py                  # Protein biological pathway annotation
│
├── data/
│   ├── raw/                         # ⛔ GITIGNORED — place kpmp_somascan.csv here
│   └── processed/                   # ⛔ GITIGNORED — auto-generated cleaned dataset
│
├── artifacts/
│   ├── column_manifest.json         # Feature inventory (clinical vs proteomic)
│   ├── cv_results.json              # Full repeated CV results per arm
│   ├── arm_comparison.csv           # Table 2 source
│   ├── calibration_metrics.json     # ECE, Brier, log loss (calibrated vs uncalibrated)
│   ├── feature_importance.csv       # XGBoost gain + permutation importance
│   ├── protein_annotation.csv       # Pathway classification of top features
│   ├── figures/
│   │   ├── fig1_calibration.png     # Reliability diagrams (uncal vs calibrated)
│   │   ├── fig2_importance.png      # Top-N feature importance bar chart
│   │   └── fig3_pathways.png        # Biological pathway stacked bar
│   └── tables/
│       └── table2_performance.csv   # Model comparison table
│
├── manuscript/
│   ├── ATI_proteomics_draft_v1.md   # Full IMRAD manuscript draft
│   ├── abstract_v1.md               # Structured abstract (248 words)
│   └── references.bib               # BibTeX (25 verified references)
│
├── governance/                      # Audit trail — RO-2026-001
│   ├── decision_log.md              # Timestamped decision rationale
│   ├── limitations_register.md      # Pre-catalogued limitations (10 items)
│   ├── novelty_statement.md         # Pre-writing novelty gate (600+ words)
│   └── run_manifest.json            # Run ID, metrics, status
│
├── tests/
│   └── test_pipeline.py             # Unit tests for preprocessing and metrics
│
├── docs/
│   └── TRIPOD_checklist.md          # TRIPOD+AI compliance checklist
│
└── .github/
    └── workflows/
        └── ci.yml                   # GitHub Actions: lint + test on push
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/fxmedus/ati-proteomics.git
cd ati-proteomics
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Place real data (after DUA approval)

```bash
cp ~/Downloads/kpmp_somascan.csv data/raw/kpmp_somascan.csv
```

### 3. Run full pipeline

```bash
# Training pipeline (preprocessing → 3 arms → calibration → feature importance)
python3 src/train.py

# Generate publication figures (300 DPI PNG)
python3 src/figures.py

# Biological pathway annotation
python3 src/annotate.py
```

### 4. Run tests

```bash
pip install pytest
pytest tests/ -v
```

### 5. Run with synthetic data (no DUA required)

If `data/raw/kpmp_somascan.csv` is absent, all scripts auto-generate a synthetic
KPMP-matched cohort (n=44) for pipeline demonstration. All outputs are labeled
`[SIMULATED]` and must not be used as empirical results.

---

## Model Architecture

| Arm | Features | Model | Purpose |
|-----|----------|-------|---------|
| Arm 1 | Clinical only (age, sex, eGFR) | L2 Logistic Regression | Physician baseline |
| Arm 2 | Proteomics only | Lasso LR (C=0.01) | Biomarker-only signal |
| Arm 3 (primary) | Clinical + Proteomics | Lasso → XGBoost → Isotonic Calibration | Best predictive system |

**Validation:** RepeatedStratifiedKFold (n_splits=5, n_repeats=10) = 50 folds  
**Primary metric:** Log loss (probabilistic sharpness)  
**Calibration:** CalibratedClassifierCV(method='isotonic', cv=5)

---

## Simulated Results [Replace with real KPMP output]

| Model | AUROC | Log Loss | Brier |
|-------|-------|----------|-------|
| Clinical only | 0.870 [0.534–1.000] | 0.463 [0.227–0.929] | 0.149 |
| Proteomics only | 0.500 [null] | 0.693 | 0.250 |
| Combined — calibrated | 0.858 [0.521–1.000] | 0.783 [0.146–4.762] | 0.155 |

> All values from synthetic cohort. ECE calibrated: 0.421 vs uncalibrated: 0.395.

---

## Governance

This repository implements the Frontier Translational Research Lab governance standards:

| Standard | Implementation |
|----------|----------------|
| Planning Before Acting | Research Order RO-2026-001 issued before any code execution |
| Novelty gate | `governance/novelty_statement.md` completed before Introduction draft |
| Limitations register | `governance/limitations_register.md` completed before Discussion draft |
| Decision audit trail | `governance/decision_log.md` — all analytic decisions documented with rationale |
| Run manifest | `governance/run_manifest.json` — reproducible run ID and parameter snapshot |
| Reporting standard | TRIPOD+AI (Collins et al. *BMJ* 2024) — see `docs/TRIPOD_checklist.md` |
| No fabricated citations | All 25 references verified via PubMed/DOI before inclusion |
| No overclaiming | All biological interpretations framed as hypothesis-generating |

---

## Citation

If you use this code or pipeline, please cite:

```
Borges J. Calibrated plasma proteomic classification of acute tubular injury:
a machine learning analysis of the Kidney Precision Medicine Project.
[Manuscript in preparation]. Frontier Translational Research Lab, 2026. https://github.com/fxmedus/ati-proteomics
```

And the anchor dataset:

```
Schmidt IM, Surapaneni AL, Zhao R, et al. Plasma proteomics of acute tubular injury.
Nat Commun. 2024;15(1):7368. doi:10.1038/s41467-024-51304-x
```

---

## License

MIT License. See [LICENSE](LICENSE).  
Data use is governed by the KPMP Data Use Agreement — see <https://www.kpmp.org/available-data>.

---

<div align="center">

**Frontier Translational Research Lab**

Department of Computer Science · Boston University · Harvard Medical School GCSRT Alumni

[![Lab Website](https://img.shields.io/badge/Lab-frontier--lab-002244?style=flat-square)](https://julian-borges-md.github.io/frontier-lab/)
[![BU CS](https://img.shields.io/badge/BU-Computer_Science-cc0000?style=flat-square)](https://www.bu.edu/cs/)
[![HMS](https://img.shields.io/badge/HMS-GCSRT_Alumni-a51c30?style=flat-square)](https://ghsm.hms.harvard.edu/education/global-clinical-scholars-research-training)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--9929--3135-a6ce39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0009-0001-9929-3135)
[![CV](https://img.shields.io/badge/Academic_CV-research--profile-4f46e5?style=flat-square)](https://julian-borges-md.github.io/research-profile/)

*Julian Borges, MD, MS · jyborges@bu.edu*

</div>
