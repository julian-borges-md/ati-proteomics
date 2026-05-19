# TRIPOD+AI Compliance Checklist
# Calibrated Plasma Proteomic Classification of Acute Tubular Injury
# RO-2026-001 | Frontier Translational Research Lab

**Reporting standard:** Collins GS, Moons KGM, Dhiman P, et al.
TRIPOD+AI statement: updated guidance for reporting clinical prediction
models that use regression or machine learning methods.
*BMJ.* 2024;385:e078378. DOI: 10.1136/bmj-2023-078378

**Study type:** Development of a prediction model (no external validation)
**Model type:** Machine learning (XGBoost + isotonic calibration)
**Checklist version:** TRIPOD+AI 2024 (27-item; supersedes TRIPOD 2015)

---

## Section 1 — Title and Abstract

| Item | Requirement | Status | Manuscript Location |
|------|-------------|--------|---------------------|
| 1 | Identify study as developing/validating a prediction model; specify ML method | COMPLETE | Title; Abstract Methods |
| 2 | Structured abstract per TRIPOD+AI for Abstracts checklist | COMPLETE | manuscript/abstract_v1.md |

---

## Section 2 — Introduction

| Item | Requirement | Status | Manuscript Location |
|------|-------------|--------|---------------------|
| 3a | Medical context and rationale for prediction model | COMPLETE | Introduction ¶1–2 |
| 3b | Intended use and target population | COMPLETE | Introduction ¶4–5 |
| 3c | Objectives of the study | COMPLETE | Introduction ¶5 |

---

## Section 3 — Methods

| Item | Requirement | Status | Manuscript Location |
|------|-------------|--------|---------------------|
| 4a | Study design; data source | COMPLETE | Methods 2.1 |
| 4b | Eligibility criteria; setting and dates | COMPLETE | Methods 2.1 |
| 5 | Outcome definition; how and when assessed | COMPLETE | Methods 2.1 (biopsy-confirmed ATI) |
| 6 | Candidate predictors; handling of missing data | COMPLETE | Methods 2.2–2.3 |
| 7 | Sample size and justification | COMPLETE | Methods 2.4 (n=44; limitation acknowledged) |
| 8 | How missing predictor data were handled | COMPLETE | Methods 2.2 (median imputation) |
| 9 | ML method(s) specified with hyperparameters | COMPLETE | Methods 2.4 (all params in code) |
| 10a | Feature selection procedure | COMPLETE | Methods 2.3 (Lasso SelectFromModel) |
| 10b | Internal validation strategy | COMPLETE | Methods 2.4 (RepeatedStratifiedKFold) |
| 10c | External validation strategy | N/A | Not performed (n=44 constraint) |
| 11 | Measures of model performance | COMPLETE | Methods 2.4 (log loss primary; AUROC, AUPRC, Brier, ECE secondary) |
| 12 | Calibration approach | COMPLETE | Methods 2.5 (isotonic regression; ECE; reliability diagram) |

---

## Section 4 — Results

| Item | Requirement | Status | Manuscript Location |
|------|-------------|--------|---------------------|
| 13 | Participant flow | COMPLETE | Results 3.1 |
| 14 | Characteristics of participants | COMPLETE | Results 3.1; Table 1 |
| 15 | Number of events and outcome rates | COMPLETE | Results 3.1 (n=24/44 ATI positive) |
| 16 | Model performance metrics with uncertainty | COMPLETE | Results 3.2; Table 2 (95% CI) |
| 17 | Calibration results | COMPLETE | Results 3.3; Figure 1 |
| 18 | Results of internal validation | COMPLETE | Results 3.2 (50-fold CV) |
| 19 | Feature importance | COMPLETE | Results 3.4; Figure 2 |

---

## Section 5 — Discussion

| Item | Requirement | Status | Manuscript Location |
|------|-------------|--------|---------------------|
| 20 | Overall interpretation; clinical implications | COMPLETE | Discussion 4.1, 4.4 |
| 21 | Comparison to prior work | COMPLETE | Discussion 4.2 |
| 22 | Limitations | COMPLETE | Discussion 4.5 (10 items; governance/limitations_register.md) |

---

## Section 6 — Other

| Item | Requirement | Status | Manuscript Location |
|------|-------------|--------|---------------------|
| 23 | Supplementary materials (code, data) | COMPLETE | GitHub repo; Zenodo DOI (pending acceptance) |
| 24 | Funding and conflicts of interest | COMPLETE | Title page |
| 25 | Ethical approval / data use | COMPLETE | Methods 2.1 (KPMP DUA) |
| 26 | Authorship | IN PROGRESS | Title page placeholder |
| 27 | Data and code availability statement | COMPLETE | Title page; Methods 2.7 |

---

## Summary

| Status | Count |
|--------|-------|
| COMPLETE | 24 |
| N/A (external validation not performed) | 1 |
| IN PROGRESS | 1 |
| MISSING | 0 |

**Outstanding action:** Finalize co-author list (Item 26) before submission.

---

*Checklist completed: 2026-04-12*
*Completed by: Julian Borges, MD*
*Governance file: RO-2026-001*
