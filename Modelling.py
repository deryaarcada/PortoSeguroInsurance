# modeling.py
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def run_lgbm_cv(X, y):

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 3,
        "min_child_samples": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        gini = 2 * auc - 1
        aucs.append(auc)

        print(f"Fold {fold}: AUC={auc:.5f}, Gini={gini:.5f}")

    mean_auc = np.mean(aucs)
    mean_gini = 2 * mean_auc - 1

    print("\n======================")
    print(f"CV Mean AUC : {mean_auc:.5f}")
    print(f"CV Mean Gini: {mean_gini:.5f}")
    print("======================")

    return mean_auc, mean_gini
