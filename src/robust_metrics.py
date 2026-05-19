#!/usr/bin/env python3
"""
robust_metrics.py — Robustness and sensitivity analysis for ATI proteomics pipeline
====================================================================================
RO-2026-001 | Frontier Translational Research Lab | Boston University
PI: Julian Borges, MD | jyborges@bu.edu | ORCID: 0009-0001-9929-3135

Computes from out-of-fold predictions (artifacts/predictions_combined.csv):
  1. Bootstrap 95% CIs on AUROC and AUPRC (n=2000 resamples)
  2. Label-permutation test on AUROC (n=1000 shuffles)
  3. AKI-only subgroup sensitivity analysis (n=61, ATI+=28)
  4. ECE from calibration curve (5-bin quantile)

Outputs:
  artifacts/run_manifest.json          — updated manifest with all robust metrics
  artifacts/run_manifest_robust.json   — same (robustness-specific copy)

USAGE
-----
    python3 src/robust_metrics.py
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score
from sklearn.calibration import calibration_curve

ART='artifacts'
preds=pd.read_csv(f'{ART}/predictions_combined.csv')
p1=preds['prob_clinical_lr'].values; p2=preds['prob_xgb_calibrated'].values
y=preds['ATI'].values; enroll=preds['Enrollment Category'].values
print(f"n={len(y)} ATI+={y.sum()} prev={y.mean():.1%}")

def bci(yt,yp,fn,nb=2000):
    rng=np.random.default_rng(42); s=[]
    for _ in range(nb):
        i=rng.integers(0,len(yt),len(yt))
        if len(np.unique(yt[i]))<2: continue
        s.append(fn(yt[i],yp[i]))
    return np.mean(s),np.percentile(s,2.5),np.percentile(s,97.5)

print("\nBootstrap CIs...")
am1,al1,ah1=bci(y,p1,roc_auc_score); ap1m,ap1l,ap1h=bci(y,p1,average_precision_score)
am2,al2,ah2=bci(y,p2,roc_auc_score); ap2m,ap2l,ap2h=bci(y,p2,average_precision_score)
print(f"  LR:  AUROC {am1:.3f} [{al1:.3f}-{ah1:.3f}] AUPRC {ap1m:.3f} [{ap1l:.3f}-{ap1h:.3f}]")
print(f"  XGB: AUROC {am2:.3f} [{al2:.3f}-{ah2:.3f}] AUPRC {ap2m:.3f} [{ap2l:.3f}-{ap2h:.3f}]")

print("\nAKI-only subgroup...")
aki=enroll=='AKI'; ya=y[aki]; p1a=p1[aki]; p2a=p2[aki]
print(f"  n={aki.sum()} ATI+={ya.sum()}")
print(f"  LR: AUROC={roc_auc_score(ya,p1a):.3f} AUPRC={average_precision_score(ya,p1a):.3f}")
print(f"  XGB: AUROC={roc_auc_score(ya,p2a):.3f} AUPRC={average_precision_score(ya,p2a):.3f}")

print("\nPermutation test (n=1000)...")
rng=np.random.default_rng(42); nl=[]; nx=[]
for _ in range(1000):
    ys=rng.permutation(y); nl.append(roc_auc_score(ys,p1)); nx.append(roc_auc_score(ys,p2))
pv_l=(sum(np.array(nl)>=roc_auc_score(y,p1))+1)/1001
pv_x=(sum(np.array(nx)>=roc_auc_score(y,p2))+1)/1001
print(f"  LR: obs={roc_auc_score(y,p1):.3f} null={np.mean(nl):.3f} p={pv_l:.4f}")
print(f"  XGB: obs={roc_auc_score(y,p2):.3f} null={np.mean(nx):.3f} p={pv_x:.4f}")

fp,mp=calibration_curve(y,p2,n_bins=5,strategy='quantile')
ece=float(np.mean(np.abs(fp-mp)))

m={'run_timestamp':datetime.now().strftime('%Y%m%d_%H%M%S'),
   'dataset':'KPMP SomaScan 2022+2024 harmonized',
   'n_samples':320,'n_ati_positive':28,'n_ati_negative':292,'prevalence':float(y.mean()),
   'random_classifier_auprc':float(y.mean()),
   'clinical_lr':{'auroc':round(am1,3),'auroc_ci_95':[round(al1,3),round(ah1,3)],
    'auprc':round(ap1m,3),'auprc_ci_95':[round(ap1l,3),round(ap1h,3)],
    'log_loss':round(float(log_loss(y,p1)),3),'brier':round(float(brier_score_loss(y,p1)),3),
    'permutation_pval':round(pv_l,4),
    'aki_only_auroc':round(float(roc_auc_score(ya,p1a)),3),
    'aki_only_auprc':round(float(average_precision_score(ya,p1a)),3)},
   'xgb_calibrated':{'auroc':round(am2,3),'auroc_ci_95':[round(al2,3),round(ah2,3)],
    'auprc':round(ap2m,3),'auprc_ci_95':[round(ap2l,3),round(ap2h,3)],
    'log_loss':round(float(log_loss(y,p2)),3),'brier':round(float(brier_score_loss(y,p2)),3),
    'ece':round(ece,3),'permutation_pval':round(pv_x,4),
    'aki_only_auroc':round(float(roc_auc_score(ya,p2a)),3),
    'aki_only_auprc':round(float(average_precision_score(ya,p2a)),3)}}
json.dump(m,open(f'{ART}/run_manifest.json','w'),indent=2,default=str)
json.dump(m,open(f'{ART}/run_manifest_robust.json','w'),indent=2,default=str)
print(f"\nDONE. ECE={ece:.3f}")
