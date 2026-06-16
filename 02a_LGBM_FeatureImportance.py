"""
Compute LightGBM feature importances using 5-fold CV.

Behavior:
- Loads preprocessed `X_preprocessed.csv` and `y_preprocessed.csv`.
- Runs 5-fold stratified cross-validation.
- Uses native LightGBM categorical handling for ps_car_11_cat (No Target Encoding).
- Extracts feature importances from each fold and averages them.
- Saves `model_results/lgbm_feature_importance.csv` and visualization.
"""

import os
import gc
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

OUTPUT_DIR = 'model_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


LGBM_PARAMS = dict(
    n_estimators=500,        
    learning_rate=0.05,      
    num_leaves=6,            
    subsample=0.8,           
    colsample_bytree=0.8,    
    objective='binary',      
    class_weight='balanced', 
    random_state=42,         
    n_jobs=2,
    verbosity=-1
)


def load_data():
    X = pd.read_csv('X_preprocessed.csv')
    y = pd.read_csv('y_preprocessed.csv').iloc[:, 0]
    try:
        categorical_cols = joblib.load('categorical_cols.pkl')
    except Exception:
        categorical_cols = []

    rare_map_path = 'rare_class_map.pkl'
    if os.path.exists(rare_map_path):
        rare_map = joblib.load(rare_map_path)
        for col, rare_vals in rare_map.items():
            if col in X.columns:
                X[col] = X[col].replace(rare_vals, -1)
    return X, y, categorical_cols


def main():
    X, y, categorical_cols = load_data()
    feature_names = X.columns.tolist()
    
    # 5-Fold CV to compute averaged feature importances
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_importances = []
    
    print("Running 5-Fold CV for Feature Importance Extraction...\n")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_f = X.iloc[train_idx].copy()
        X_val_f = X.iloc[val_idx].copy()
        y_train_f = y.iloc[train_idx]
        y_val_f = y.iloc[val_idx]
        
        # ===== CATEGORICAL VARIABLES PREPARATION =====
        cat_feats = [c for c in categorical_cols if c in X_train_f.columns]
        for c in cat_feats:
            X_train_f[c] = X_train_f[c].astype('category')
            X_val_f[c] = X_val_f[c].astype('category')
        
        # ===== MODEL TRAINING =====
        print(f"Fold {fold}: Training LightGBM...")
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train_f, y_train_f,
            eval_set=[(X_val_f, y_val_f)],
            eval_metric='auc',
            categorical_feature=cat_feats if cat_feats else None,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50), 
                lgb.log_evaluation(period=0)
            ]
        )
        
        # ===== FEATURE IMPORTANCE SCORES =====
        fold_importance = model.feature_importances_
        fold_importances.append(fold_importance)
        print(f"Fold {fold}: Complete")
        
        del model, X_train_f, X_val_f, y_train_f, y_val_f
        gc.collect()
    
    # ===== AVERAGE THE FOLDS =====
    avg_importance = np.mean(fold_importances, axis=0)
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    # ===== SAVE RESULTS =====
    out_csv = os.path.join(OUTPUT_DIR, 'lgbm_feature_importance.csv')
    feature_importance_df.to_csv(out_csv, index=False)
    print(f'\n✓ Saved importances to: {out_csv}\n')
    
    print('Top 20 Feature Importances (LGBM - 5-Fold Averaged):')
    print(feature_importance_df.head(20).to_string(index=False))
    
    # ===== VISUALIZATION =====
    plt.figure(figsize=(6, 4))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), color='skyblue')
    plt.title("Top 20 Feature Importances (LGBM - 5-Fold Averaged)", fontsize=12, fontweight='bold')
    plt.xlabel('Importance Score (Averaged)')
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, 'lgbm_feature_importance.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved plot to: {out_png}')


if __name__ == '__main__':
    main()
