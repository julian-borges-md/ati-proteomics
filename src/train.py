"""
ATI Plasma Proteomics ML Pipeline — Real Data
RO-2026-001 | Frontier Translational Research Lab | BU
PI: Julian Borges, MD | jyborges@bu.edu
Data: KPMP SomaScan Plasma 2024 (n=184) + Open Access Clinical Data
ATI label: Primary Adjudicated Category == 'Acute Tubular Injury'
ATI positive: n=11 | ATI negative: n=173
Note: n=11 ATI — no Lasso proteomic arm; XGBoost with SMOTE + calibration only.
All results are computational. No biological validation claims.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
ART_DIR   = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(ART_DIR, exist_ok=True)

SOMA_FILE  = os.path.join(DATA_DIR, 'kpmp_somascan_2024.xlsx')
CLIN_FILE  = os.path.join(DATA_DIR, 'kpmp_clinical.csv')

# ── Clinical features to include ──────────────────────────────────────────────
CLINICAL_COLS = ['Baseline eGFR (ml/min/1.73m2)', 'KDIGO Stage']
KDIGO_MAP = {'Stage 1': 1, 'Stage 2': 2, 'Stage 3': 3}

def load_data():
    """Load and merge SomaScan + clinical data. Return X_protein, X_clinical, y, metadata."""
    print("[LOAD] Reading SomaScan XLSX (header row 4)...")
    soma = pd.read_excel(SOMA_FILE, header=4)
    print(f"  SomaScan shape: {soma.shape}")

    # Identify protein columns (Gene.SeqId format after col 14)
    meta_cols = soma.columns[:15].tolist()
    protein_cols = soma.columns[15:].tolist()
    print(f"  Protein columns: {len(protein_cols)}")

    print("[LOAD] Reading clinical data...")
    clin = pd.read_csv(CLIN_FILE)
    print(f"  Clinical shape: {clin.shape}")

    # Merge on Participant ID
    df = soma.merge(clin[['Participant ID', 'Primary Adjudicated Category',
                            'Enrollment Category', 'Sex',
                            'Age (Years) (Binned)',
                            'Baseline eGFR (ml/min/1.73m2)',
                            'KDIGO Stage', 'Race']],
                    on='Participant ID', how='left')
    print(f"  Merged shape: {df.shape}")

    # Build ATI binary label
    df['ATI'] = (df['Primary Adjudicated Category'] == 'Acute Tubular Injury').astype(int)
    print(f"\n  ATI=1: {df['ATI'].sum()} | ATI=0: {(df['ATI']==0).sum()}")

    # Protein matrix — log2 transform (standard for SomaScan RFU)
    X_protein = np.log2(df[protein_cols].astype(float).clip(lower=1))
    print(f"\n  X_protein shape: {X_protein.shape}")

    # Clinical features
    df['KDIGO_num'] = df['KDIGO Stage'].map(KDIGO_MAP).fillna(0)
    X_clin = df[['Baseline eGFR (ml/min/1.73m2)', 'KDIGO_num']].fillna(
        df[['Baseline eGFR (ml/min/1.73m2)', 'KDIGO_num']].median())

    y = df['ATI'].values
    meta = df[['Participant ID', 'Primary Adjudicated Category',
               'Enrollment Category', 'ATI']].copy()

    return X_protein, X_clin, y, meta, protein_cols


def run_arm1_clinical_lr(X_clin, y):
    """Arm 1: Clinical LR baseline (eGFR + KDIGO)."""
    print("\n[ARM 1] Clinical LR baseline...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('lr', LogisticRegression(class_weight='balanced',
                                               max_iter=1000, random_state=42))])
    probs = cross_val_predict(pipe, X_clin, y, cv=cv, method='predict_proba')[:,1]
    auroc = roc_auc_score(y, probs)
    ll    = log_loss(y, probs)
    bs    = brier_score_loss(y, probs)
    print(f"  AUROC={auroc:.3f}  LogLoss={ll:.3f}  Brier={bs:.3f}")
    return {'arm': 'clinical_lr', 'auroc': auroc, 'log_loss': ll, 'brier': bs, 'probs': probs.tolist()}


def run_arm2_xgb_calibrated(X_protein, X_clin, y, protein_cols):
    """Arm 2: XGBoost on top-variance proteins + clinical, with Platt calibration.
       Given n=11 ATI, use scale_pos_weight for imbalance; no SMOTE (n too small).
       Feature selection: top-500 proteins by variance to reduce dimensionality.
    """
    print("\n[ARM 2] XGBoost (top-500 proteins + clinical), calibrated...")
    X_p = np.array(X_protein)
    X_c = np.array(X_clin)

    # Select top-500 proteins by variance
    variances = X_p.var(axis=0)
    top_idx = np.argsort(variances)[::-1][:500]
    X_top = X_p[:, top_idx]
    top_protein_names = [protein_cols[i] for i in top_idx]
    print(f"  Using top-500 proteins by variance")

    # Combined feature matrix
    X_combined = np.hstack([X_top, X_c])

    scale_pos = (y == 0).sum() / (y == 1).sum()
    print(f"  scale_pos_weight={scale_pos:.1f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                        scale_pos_weight=scale_pos,
                        use_label_encoder=False, eval_metric='logloss',
                        random_state=42, verbosity=0)

    probs_raw = cross_val_predict(xgb, X_combined, y, cv=cv, method='predict_proba')[:,1]
    auroc_raw = roc_auc_score(y, probs_raw)
    ll_raw    = log_loss(y, probs_raw)
    bs_raw    = brier_score_loss(y, probs_raw)
    print(f"  Uncalibrated: AUROC={auroc_raw:.3f}  LogLoss={ll_raw:.3f}  Brier={bs_raw:.3f}")

    # Platt calibration via CalibratedClassifierCV
    cal_xgb = CalibratedClassifierCV(xgb, method='sigmoid', cv=3)
    probs_cal = cross_val_predict(cal_xgb, X_combined, y, cv=cv, method='predict_proba')[:,1]
    auroc_cal = roc_auc_score(y, probs_cal)
    ll_cal    = log_loss(y, probs_cal)
    bs_cal    = brier_score_loss(y, probs_cal)

    # Expected Calibration Error (ECE)
    fraction_pos, mean_pred = calibration_curve(y, probs_cal, n_bins=5, strategy='quantile')
    ece = np.mean(np.abs(fraction_pos - mean_pred))
    print(f"  Calibrated:   AUROC={auroc_cal:.3f}  LogLoss={ll_cal:.3f}  Brier={bs_cal:.3f}  ECE={ece:.3f}")

    return {
        'arm': 'xgb_calibrated',
        'auroc_uncal': auroc_raw, 'log_loss_uncal': ll_raw, 'brier_uncal': bs_raw,
        'auroc_cal': auroc_cal, 'log_loss_cal': ll_cal, 'brier_cal': bs_cal,
        'ece': ece,
        'n_proteins': 500,
        'top_proteins': top_protein_names[:20],
        'probs_cal': probs_cal.tolist()
    }


def save_results(results, meta, y):
    """Save all artifacts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Run manifest
    manifest = {
        'run_timestamp': timestamp,
        'n_samples': int(len(y)),
        'n_ati_positive': int(y.sum()),
        'n_ati_negative': int((y==0).sum()),
        'prevalence': float(y.mean()),
        'data_source': 'KPMP SomaScan Plasma 2024 + OpenAccessClinicalData 20260309',
        'kpmp_citation': 'Accessed April 12, 2026. https://www.kpmp.org',
        'arms': [r['arm'] for r in results]
    }
    for r in results:
        manifest[r['arm']] = {k: v for k, v in r.items()
                               if k not in ['probs', 'probs_cal', 'top_proteins']}

    manifest_path = os.path.join(ART_DIR, f'run_manifest_{timestamp}.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n[SAVE] Manifest: {manifest_path}")

    # Per-sample predictions
    pred_df = meta.copy()
    pred_df['y_true'] = y
    for r in results:
        key = 'probs_cal' if 'probs_cal' in r else 'probs'
        if key in r:
            pred_df[f'prob_{r["arm"]}'] = r[key]
    pred_path = os.path.join(ART_DIR, f'predictions_{timestamp}.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"[SAVE] Predictions: {pred_path}")

    # Also save latest manifest as fixed name for governance
    with open(os.path.join(ART_DIR, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest_path, pred_path


def main():
    print("=" * 60)
    print("ATI Plasma Proteomics ML Pipeline — REAL DATA RUN")
    print("RO-2026-001 | Frontier Translational Research Lab")
    print("=" * 60)

    X_protein, X_clin, y, meta, protein_cols = load_data()

    results = []
    results.append(run_arm1_clinical_lr(X_clin, y))
    results.append(run_arm2_xgb_calibrated(X_protein, X_clin, y, protein_cols))

    manifest_path, pred_path = save_results(results, meta, y)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  ATI n={int(y.sum())} vs n={int((y==0).sum())}")
    for r in results:
        arm = r['arm']
        if 'auroc_cal' in r:
            print(f"  {arm}: AUROC={r['auroc_cal']:.3f}  ECE={r['ece']:.3f}")
        else:
            print(f"  {arm}: AUROC={r['auroc']:.3f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
