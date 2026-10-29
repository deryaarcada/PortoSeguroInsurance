import joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score
from metrics import calculate_ece
from preprocessing_utils import apply_rare_map

bundle = joblib.load("model_results/calibrated_predictor.pkl")
X = pd.read_csv("X_preprocessed.csv")
y = pd.read_csv("y_preprocessed.csv").iloc[:, 0].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

raw = np.zeros(len(X)); cal = np.zeros(len(X))
for fm, (tr, va) in zip(bundle["fold_models"], skf.split(X, y)):
    Xv = apply_rare_map(X.iloc[va], fm["rare_map"])[bundle["feature_names"]].copy()
    for c in bundle["categorical_features"]:
        Xv[c] = Xv[c].astype("category")
    p = fm["model"].predict_proba(Xv)[:, 1]
    raw[va] = p
    cal[va] = fm["iso_reg"].transform(p)

for name, p in [("raw", raw), ("calibrated", cal)]:
    print(f"{name:11s} Gini={2*roc_auc_score(y,p)-1:.5f}  Brier={brier_score_loss(y,p):.5f}  "
          f"ECE={calculate_ece(y,p):.5f}  mean pred={p.mean():.4f}")

order = np.argsort(cal)[::-1]
for k in [0.005, 0.01, 0.02]:
    idx = order[:int(len(cal) * k)]
    print(f"top {k:.1%}: predicted={cal[idx].mean():.4f}  observed={y[idx].mean():.4f}")

print(f"observed claim rate = {y.mean():.4f}")

for name, p in [("raw", raw), ("calibrated", cal)]:
    fold_ece = [calculate_ece(y[va], p[va]) for _, va in skf.split(X, y)]
    print(f"{name:11s} mean fold ECE = {np.mean(fold_ece):.5f}")