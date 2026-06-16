"""
PHASE 2c: Hyperparameter Tuning via Optuna (Maximizing Gini Index)
================================================================
This script finds the optimal hyperparameters for LightGBM using
Bayesian Optimization (TPE) with 5-Fold Stratified Cross-Validation.

Requirements: pip install optuna
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

try:
    import optuna
except ImportError:
    raise ImportError("Optuna library is missing. 'pip install optuna'")

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUTPUT_DIR = 'model_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_data():
    print('Loading preprocessed data for hyperparameter tuning...')
    X = pd.read_csv('X_preprocessed.csv')
    y = pd.read_csv('y_preprocessed.csv').iloc[:, 0]
    categorical_cols = joblib.load('categorical_cols.pkl')

    rare_map_path = 'rare_class_map.pkl'
    if os.path.exists(rare_map_path):
        rare_class_map = joblib.load(rare_map_path)
        for col, rare_vals in rare_class_map.items():
            if col in X.columns:
                X[col] = X[col].replace(rare_vals, -1)
    
    # index issue solution
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Convert all categorical variables to 'category' type
    cat_feats = [c for c in categorical_cols if c in X.columns]
    for col in cat_feats:
        X[col] = X[col].astype('category')
        
    return X, y, cat_feats


def objective(trial, X, y, categorical_features):
    # Hyperparameter ranges that Optuna will test
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 400, 1200, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 5, 15),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 100, 1000, step=100),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),  
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), 
        'objective': 'binary',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_ginis = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            categorical_feature=categorical_features if categorical_features else None,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        gini = 2 * auc - 1
        fold_ginis.append(gini)

    return np.mean(fold_ginis)


def main():
    X, y, cat_feats = load_data()
    
    print(f"\nData shapes successfully realigned -> X: {X.shape}, y: {y.shape}")
    print("\nStarting Bayesian Optimization via Optuna (30 Trials)...")
    print("This will evaluate 30 different parameter combinations using 5-Fold CV.")
    
    study = optuna.create_study(direction='maximize')
    
    def logging_callback(study, trial):
        if trial.value is not None:
            print(f"Trial {trial.number:02d}/30 | Current Trial Gini: {trial.value:.5f} | Best Gini So Far: {study.best_value:.5f}")
        else:
            print(f"Trial {trial.number:02d}/30 | Current Trial Failed (Value None) | Best Gini So Far: {study.best_value:.5f}")

    # The `catch=(Exception,)` parameter prevents the entire process from crashing in case of any error.
    study.optimize(
        lambda trial: objective(trial, X, y, cat_feats), 
        n_trials=30, 
        callbacks=[logging_callback],
        catch=(Exception,)
    )

    print('\n' + '=' * 70)
    print('HYPERPARAMETER TUNING COMPLETED')
    print('=' * 70)
    
    try:
        print(f"Best 5-Fold CV Gini Index: {study.best_value:.6f}")
        print("\nBest Hyperparameters Found:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.5f}")
            else:
                print(f"  {key}: {value}")
                
        best_params_path = os.path.join(OUTPUT_DIR, 'best_lgbm_params.pkl')
        joblib.dump(study.best_params, best_params_path)
        print(f"\n✓ Best parameters saved to: {best_params_path}")
    except ValueError:
        print("All attempts failed. Please check the structure of the datasets.")


if __name__ == '__main__':
    main()
