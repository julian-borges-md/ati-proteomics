# Decision Log — RO-2026-001
**Project:** ATI Plasma Proteomics Prediction Model
**Run ID:** RO-2026-001-A
**Start timestamp:** 2026-04-12T00:00:00Z

---

| Timestamp | Decision | Rationale |
|-----------|----------|-----------|
| 2026-04-12T00:01 | Dataset: KPMP SomaScan (n=44, Schmidt et al. 2024) selected as primary | Per RO specification; credentialed public access; DUA required for download |
| 2026-04-12T00:02 | Simulation protocol activated | KPMP DUA pending; synthetic cohort from published parameters used for pipeline demonstration; all outputs labeled SIMULATED |
| 2026-04-12T00:03 | Three model arms per RO Section 3.2 | Clinical-only LR, proteomics-only Lasso, combined Lasso+XGBoost+CalibratedCV |
| 2026-04-12T00:04 | Calibration method: isotonic regression via CalibratedClassifierCV | Per RO 3.3; isotonic preferred over Platt scaling for small-N nonlinear calibration |
| 2026-04-12T00:05 | Feature selection threshold: mean (SelectFromModel default) | Per RO 3.2 Arm 3 specification |
| 2026-04-12T00:06 | Validation: RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42) | Per RO 3.3; small N requires repeated CV to stabilize variance |
| 2026-04-12T00:07 | TRIPOD+AI (2024) selected over TRIPOD 2015 | Collins et al. 2024 BMJ: TRIPOD+AI supersedes 2015 checklist for ML studies |
| 2026-04-12T00:08 | Primary journal: Clinical Kidney Journal (OUP) | Per RO Section 4.1; nephrology focus, open access |
| 2026-04-12T00:09 | Biological annotation: 7 pathway categories per RO Section 3.6 | Assigned from UniProt and ATI literature (Schmidt et al. 2024) |
| 2026-04-12T00:10 | All performance results labeled [SIMULATED] | Ethical requirement: no fabricated empirical claims; real results replace on DUA approval |
| 2026-04-12T00:11 | GitHub repo structure adopted: src/, data/, artifacts/, governance/, manuscript/, tests/, docs/ | FAIR data principles; reproducibility standard; CI via GitHub Actions |
| 2026-04-12T00:12 | .gitignore excludes data/raw/ and data/processed/ | KPMP DUA prohibits redistribution of raw data |
