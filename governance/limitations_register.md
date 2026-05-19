# Limitations Register — RO-2026-001
**Status:** COMPLETE — pre-writing gate (must precede Discussion draft)
**Date:** 2026-04-12

---

| # | Limitation | Category | Manuscript Location | Mitigation |
|---|-----------|----------|-------------------|------------|
| L1 | Sample size n=44 | Statistical | Methods 2.7; Discussion 4.5 | Repeated CV (50 folds); wide CIs reported; all claims framed as exploratory |
| L2 | No external validation | Generalizability | Discussion 4.5 | Internal CV only; external validation as primary future direction |
| L3 | Cross-sectional design | Causal inference | Discussion 4.5 | Diagnostic classification only; no temporal inference claimed |
| L4 | KPMP biopsy enrichment | Selection bias | Discussion 4.5 | Results may not generalize to unselected hospital populations; acknowledged |
| L5 | No independent test set | Overfitting risk | Methods 2.4; Discussion 4.5 | CalibratedClassifierCV partially corrects overconfident probabilities |
| L6 | SomaScan aptamer cross-reactivity | Measurement | Methods 2.2 | Platform-level limitation; applies equally to all SomaScan studies |
| L7 | Missing clinical covariates | Data completeness | Methods 2.2 | Median imputation applied; sensitivity analysis recommended |
| L8 | Lasso instability in small-N high-dimensional setting | Methodological | Methods 2.4 | Hyperparameter optimization (C, penalty type) required with real data |
| L9 | Pathway annotation not from GSEA | Interpretability | Methods 2.6; Discussion 4.3 | Categories are descriptive and hypothesis-generating only |
| L10 | KPMP demographic composition | Equity | Discussion 4.5 | Generalizability across populations requires multi-site validation |
