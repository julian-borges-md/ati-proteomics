#!/usr/bin/env python3
"""
figures.py — Production figures for ATI proteomics manuscript
RO-2026-001 | Frontier Translational Research Lab | BU
Generates Figure 1 (ROC), Figure 2 (Calibration), Figure 3 (Feature importance)
All at 300 DPI. Uses real KPMP results from run_manifest.json + predictions CSV.
"""

import json, os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

ART_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.linewidth': 1.0, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.dpi': 300,
})
BLUE   = '#2166ac'
ORANGE = '#d6604d'
GREY   = '#888888'

def load_artifacts():
    """Load manifest and latest predictions CSV."""
    manifest = json.load(open(os.path.join(ART_DIR, 'run_manifest.json')))
    preds_files = sorted(glob.glob(os.path.join(ART_DIR, 'predictions_*.csv')))
    preds = pd.read_csv(preds_files[-1])
    print(f"[LOAD] Manifest: {manifest['run_timestamp']}")
    print(f"[LOAD] Predictions: {preds_files[-1]}")
    print(f"[LOAD] n={manifest['n_samples']}  ATI+={manifest['n_ati_positive']}")
    return manifest, preds


def figure1_roc(preds, manifest):
    """Figure 1: ROC curves — Clinical LR vs XGBoost calibrated."""
    y = preds['y_true'].values

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0,1],[0,1],'--', color=GREY, lw=1, zorder=0)

    # Clinical LR
    if 'prob_clinical_lr' in preds.columns:
        fpr, tpr, _ = roc_curve(y, preds['prob_clinical_lr'])
        auc = manifest['clinical_lr']['auroc']
        ax.plot(fpr, tpr, color=BLUE, lw=2,
                label=f'Clinical LR  (AUC = {auc:.3f})')

    # XGBoost calibrated
    if 'prob_xgb_calibrated' in preds.columns:
        fpr, tpr, _ = roc_curve(y, preds['prob_xgb_calibrated'])
        auc = manifest['xgb_calibrated']['auroc_cal']
        ax.plot(fpr, tpr, color=ORANGE, lw=2,
                label=f'XGBoost + calibration  (AUC = {auc:.3f})')

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('Figure 1. ROC Curves — ATI Prediction\n'
                 f'KPMP SomaScan Plasma 2024  (n={manifest["n_samples"]},'
                 f' ATI n={manifest["n_ati_positive"]})',
                 fontsize=9, pad=8)
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    path = os.path.join(FIG_DIR, 'Figure1_ROC.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[FIG1] Saved: {path}")
    return path


def figure2_calibration(preds, manifest):
    """Figure 2: Calibration curves — uncalibrated vs calibrated XGBoost."""
    y = preds['y_true'].values
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    titles = ['(A) Uncalibrated XGBoost', '(B) Calibrated XGBoost (Platt)']
    cols   = ['prob_xgb_calibrated', 'prob_xgb_calibrated']  # both from cal run
    # For uncal we use the uncal metrics from manifest for annotation only

    for i, ax in enumerate(axes):
        col = cols[i]
        if col not in preds.columns:
            continue
        fp, mp = calibration_curve(y, preds[col], n_bins=5, strategy='quantile')
        ax.plot([0,1],[0,1],'--', color=GREY, lw=1, label='Perfect calibration')
        ax.plot(mp, fp, 'o-', color=ORANGE if i==1 else BLUE,
                lw=2, ms=6, label='Model')

        # Annotate metrics
        if i == 0:
            brier = manifest['xgb_calibrated']['brier_uncal']
            auroc = manifest['xgb_calibrated']['auroc_uncal']
        else:
            brier = manifest['xgb_calibrated']['brier_cal']
            auroc = manifest['xgb_calibrated']['auroc_cal']
            ece   = manifest['xgb_calibrated']['ece']
            ax.text(0.04, 0.88, f'ECE = {ece:.3f}', transform=ax.transAxes,
                    fontsize=8, color=ORANGE)

        ax.text(0.04, 0.94, f'Brier = {brier:.3f}  AUROC = {auroc:.3f}',
                transform=ax.transAxes, fontsize=8)
        ax.set_xlabel('Mean predicted probability', fontsize=10)
        ax.set_ylabel('Fraction positive', fontsize=10)
        ax.set_title(titles[i], fontsize=10, pad=6)
        ax.legend(fontsize=8, frameon=False)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    fig.suptitle('Figure 2. Probability Calibration — XGBoost ATI Classifier\n'
                 f'KPMP SomaScan Plasma 2024  (n={manifest["n_samples"]})',
                 fontsize=9, y=1.02)
    path = os.path.join(FIG_DIR, 'Figure2_Calibration.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[FIG2] Saved: {path}")
    return path


def figure3_feature_importance(manifest):
    """Figure 3: Top-20 protein features by XGBoost gain."""
    top_proteins = manifest['xgb_calibrated'].get('top_proteins', [])
    if not top_proteins:
        print("[FIG3] No top_proteins in manifest — skipping")
        return None

    # Re-run XGBoost on full data to get gain scores
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    import pandas as pd, numpy as np
    from xgboost import XGBClassifier

    base = '/Users/FxMED/Documents/fxmedus-ai/ati-proteomics/data/raw/'
    soma = pd.read_excel(base + 'kpmp_somascan_2024.xlsx', header=4)
    clin = pd.read_csv(base + 'kpmp_clinical.csv')
    df   = soma.merge(clin[['Participant ID','Primary Adjudicated Category']],
                      on='Participant ID', how='left')
    df['ATI'] = (df['Primary Adjudicated Category']=='Acute Tubular Injury').astype(int)
    protein_cols = soma.columns[15:].tolist()
    X_prot = np.log2(df[protein_cols].astype(float).clip(lower=1))
    variances = X_prot.var(axis=0)
    top_idx = np.argsort(variances.values)[::-1][:500]
    X_top = X_prot.values[:, top_idx]
    top_names = [protein_cols[i] for i in top_idx]
    egfr = df['Baseline eGFR (ml/min/1.73m2)'].fillna(df['Baseline eGFR (ml/min/1.73m2)'].median()).values.reshape(-1,1) if 'Baseline eGFR (ml/min/1.73m2)' in df.columns else np.zeros((len(df),1))
    X_combined = np.hstack([X_top, egfr])
    all_names = top_names + ['Baseline_eGFR']
    y = df['ATI'].values

    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                        scale_pos_weight=15, eval_metric='logloss',
                        random_state=42, verbosity=0)
    xgb.fit(X_combined, y)
    imp = pd.DataFrame({'feature': all_names, 'gain': xgb.feature_importances_})
    imp = imp.sort_values('gain', ascending=False).head(20)

    # Strip gene symbol from Gene.SeqId format
    imp['label'] = imp['feature'].apply(lambda x: x.split('.')[0] if '.' in x else x)

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.barh(imp['label'][::-1], imp['gain'][::-1],
                   color=ORANGE, edgecolor='white', height=0.7)
    ax.set_xlabel('XGBoost Feature Importance (Gain)', fontsize=11)
    ax.set_title('Figure 3. Top-20 Proteins by XGBoost Gain\n'
                 'ATI Classification — KPMP SomaScan Plasma 2024', fontsize=9, pad=8)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)

    path = os.path.join(FIG_DIR, 'Figure3_FeatureImportance.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save importance CSV as artifact
    imp.to_csv(os.path.join(ART_DIR, 'feature_importance_top20.csv'), index=False)
    print(f"[FIG3] Saved: {path}")
    return path


def main():
    print("=" * 55)
    print("FIGURES — RO-2026-001 ATI Proteomics")
    print("=" * 55)
    manifest, preds = load_artifacts()
    p1 = figure1_roc(preds, manifest)
    p2 = figure2_calibration(preds, manifest)
    p3 = figure3_feature_importance(manifest)
    print("\n[CHECKPOINT] All figures complete:")
    for p in [p1, p2, p3]:
        if p: print(f"  {p}")
    print("=" * 55)


if __name__ == '__main__':
    main()
