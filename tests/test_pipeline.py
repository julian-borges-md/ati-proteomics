"""
test_pipeline.py — Unit tests for ATI proteomics pipeline
==========================================================
RO-2026-001 | Frontier Translational Research Lab

Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_df():
    np.random.seed(42)
    n = 44
    labels = np.array([1]*24 + [0]*20)
    df = pd.DataFrame({
        "age":  np.random.normal(60, 14, n).clip(20, 90),
        "sex":  np.random.binomial(1, 0.55, n),
        "egfr": np.random.normal(45, 20, n).clip(5, 120),
        **{f"PROT_{i:04d}": np.random.randn(n) for i in range(50)},
        "ati_label": labels
    })
    return df


# ─── Preprocessing tests ───────────────────────────────────────────────────

def test_dataframe_shape(synthetic_df):
    assert synthetic_df.shape == (44, 54)  # 3 clin + 50 prot + label


def test_ati_prevalence(synthetic_df):
    prev = synthetic_df["ati_label"].mean()
    assert 0.40 <= prev <= 0.70, f"Unexpected prevalence: {prev}"


def test_no_negative_egfr(synthetic_df):
    assert (synthetic_df["egfr"] > 0).all()


def test_median_imputation():
    from sklearn.impute import SimpleImputer
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
    imp = SimpleImputer(strategy="median")
    out = imp.fit_transform(X)
    assert not np.any(np.isnan(out))


def test_variance_threshold():
    from sklearn.feature_selection import VarianceThreshold
    X = np.array([[1, 2, 0], [3, 4, 0], [5, 6, 0]])
    vt = VarianceThreshold(threshold=1e-8)
    out = vt.fit_transform(X)
    assert out.shape[1] == 2  # constant column removed


# ─── Model tests ───────────────────────────────────────────────────────────

def test_logistic_regression_fits(synthetic_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(
        synthetic_df[["age", "sex", "egfr"]].values)
    y = synthetic_df["ati_label"].values
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    assert hasattr(model, "coef_")


def test_xgboost_produces_probabilities(synthetic_df):
    from xgboost import XGBClassifier
    X = synthetic_df[["age", "sex", "egfr"]].values
    y = synthetic_df["ati_label"].values
    model = XGBClassifier(n_estimators=10, verbosity=0,
                          use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert probs.min() >= 0.0 and probs.max() <= 1.0


# ─── Metric tests ──────────────────────────────────────────────────────────

def test_ece_perfect_calibration():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    assert ece < 0.01


def test_log_loss_range():
    from sklearn.metrics import log_loss
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.7, 0.3])
    ll = log_loss(y_true, y_prob)
    assert 0.0 < ll < 1.0
