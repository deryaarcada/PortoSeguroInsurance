# modeling.py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


def run_cv(X, y, cat_feature_indices, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs_lgb, aucs_xgb, aucs_cb, aucs_ens = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # -------------------------
        # LightGBM
        # -------------------------
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=3,
            min_child_samples=200,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        lgb_model.fit(X_tr, y_tr)
        pred_lgb = lgb_model.predict_proba(X_val)[:, 1]

        # -------------------------
        # XGBoost
        # -------------------------
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1
        )

        xgb_model.fit(X_tr, y_tr)
        pred_xgb = xgb_model.predict_proba(X_val)[:, 1]

        # -------------------------
        # CatBoost
        # -------------------------
        cb_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=3,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False
        )

        cb_model.fit(
            X_tr,
            y_tr,
            cat_features=cat_feature_indices
        )

        pred_cb = cb_model.predict_proba(X_val)[:, 1]

        # -------------------------
        # Scores
        # -------------------------
        auc_lgb = roc_auc_score(y_val, pred_lgb)
        auc_xgb = roc_auc_score(y_val, pred_xgb)
        auc_cb  = roc_auc_score(y_val, pred_cb)

        # Ensemble (CatBoost ağırlıklı)
        pred_ens = 0.3 * pred_lgb + 0.3 * pred_xgb + 0.4 * pred_cb
        auc_ens = roc_auc_score(y_val, pred_ens)

        aucs_lgb.append(auc_lgb)
        aucs_xgb.append(auc_xgb)
        aucs_cb.append(auc_cb)
        aucs_ens.append(auc_ens)

        print(
            f"Fold {fold} | "
            f"LGB: {auc_lgb:.4f} | "
            f"XGB: {auc_xgb:.4f} | "
            f"CB: {auc_cb:.4f} | "
            f"ENS: {auc_ens:.4f}"
        )

    print("\n======================")
    print(f"LGB Mean Gini : {2*np.mean(aucs_lgb)-1:.4f}")
    print(f"XGB Mean Gini : {2*np.mean(aucs_xgb)-1:.4f}")
    print(f"CB  Mean Gini : {2*np.mean(aucs_cb)-1:.4f}")
    print(f"ENS Mean Gini : {2*np.mean(aucs_ens)-1:.4f}")
    print("======================")
