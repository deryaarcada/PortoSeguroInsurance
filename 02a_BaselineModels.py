"""
2b: Model comparison for selection (ALL FEATURES & Bootstrap CI)
================================================================================
This is the baseline comparison script.It uses ALL preprocessed features for LightGBM, RandomForest,
and XGBoost. 

It computes a 95% Confidence Interval for the Gini index using Bootstrap
resampling on the out-of-fold predictions.

Run after 01_Preprocessing.py has produced preprocessed data and
Before 02b_HyperparameterTuning.py
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sklearn_auc
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False


def bootstrap_gini_ci(y_true, y_probs, n_iterations=200):
    """
    A function that calculates the Gini using the Bootstrap method for a 95% Confidence Interval (CI).
    """
    stats = []
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    for i in range(n_iterations):
        # Resampling
        y_true_resample, y_probs_resample = resample(y_true, y_probs, random_state=i)

        # Gini calculation (2 * AUC - 1)
        auc_boot = roc_auc_score(y_true_resample, y_probs_resample)
        gini_boot = 2 * auc_boot - 1
        stats.append(gini_boot)

    # %95 Confidence Interval Limits (2.5 and 97.5 percentiles)
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return np.mean(stats), lower, upper


def evaluate_ranking(y_true, y_probs, k_percent_list=[0.01]):
    results = {}
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    results['PR-AUC'] = sklearn_auc(recall, precision)
    df_temp = pd.DataFrame({'y_true': y_true, 'y_probs': y_probs}).sort_values(by='y_probs', ascending=False)
    n_total_positives = y_true.sum()
    for k in k_percent_list:
        n_cutoff = max(1, int(len(y_true) * k))
        n_positives_at_k = df_temp.iloc[:n_cutoff]['y_true'].sum()
        recall_at_k = n_positives_at_k / n_total_positives if n_total_positives > 0 else 0
        results[f'Recall@{int(k * 100)}%'] = recall_at_k
    return results


def fit_target_encoding_map(cat_train, y_train, smoothing=10):
    global_mean = y_train.mean()
    agg = y_train.groupby(cat_train).agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth_map = (counts * means + smoothing * global_mean) / (counts + smoothing)
    return smooth_map.to_dict(), global_mean


def apply_target_encoding(cat_series, enc_map, global_mean):
    return cat_series.map(enc_map).fillna(global_mean)


def load_data():
    print('Loading preprocessed data...')
    X = pd.read_csv('X_preprocessed.csv')
    y = pd.read_csv('y_preprocessed.csv').iloc[:, 0]
    categorical_cols = joblib.load('categorical_cols.pkl')

    rare_map_path = 'rare_class_map.pkl'
    if os.path.exists(rare_map_path):
        rare_class_map = joblib.load(rare_map_path)
        for col, rare_vals in rare_class_map.items():
            if col in X.columns:
                X[col] = X[col].replace(rare_vals, -1)
        print('Applied training-derived rare class mapping')

    print(f'Using ALL preprocessed features. X shape: {X.shape}, y shape: {y.shape}')

    return X, y, categorical_cols


def run_cv_for_model(name, model_factory, X, y, categorical_cols, n_splits=5):
    print('\n' + '=' * 70)
    print(f'CV STARTING FOR: {name}')
    print('=' * 70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Pooling the validation predictions of all folds for Bootstrap
    all_y_val = []
    all_val_probs = []

    fold_pr = []
    fold_recall_1 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        current_cat_features = [c for c in categorical_cols if c in X_train.columns]

        if name == 'LightGBM':
            for col in current_cat_features:
                X_train[col] = X_train[col].astype('category')
                X_val[col] = X_val[col].astype('category')

            model = model_factory()
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                categorical_feature=current_cat_features if current_cat_features else None,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
            )
        else:
            if 'ps_car_11_cat' in X_train.columns:
                enc_map, global_mean = fit_target_encoding_map(X_train['ps_car_11_cat'], y_train, smoothing=10)
                X_train['ps_car_11_cat'] = apply_target_encoding(X_train['ps_car_11_cat'], enc_map, global_mean)
                X_val['ps_car_11_cat'] = apply_target_encoding(X_val['ps_car_11_cat'], enc_map, global_mean)

            other_cats = [c for c in current_cat_features if c != 'ps_car_11_cat']
            for col in other_cats:
                X_train[col] = X_train[col].astype('category').cat.codes
                X_val[col] = X_val[col].astype('category').cat.codes

            model = model_factory()
            model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]
        ranking = evaluate_ranking(y_val, val_probs)

        fold_pr.append(ranking['PR-AUC'])
        fold_recall_1.append(ranking['Recall@1%'])

        all_y_val.extend(y_val.values)
        all_val_probs.extend(val_probs)

        auc = roc_auc_score(y_val, val_probs)
        print(f'Fold {fold} | Gini: {2*auc-1:.5f} | PR-AUC: {ranking["PR-AUC"]:.5f} | Recall@1%: {ranking["Recall@1%"]:.5f}')

    print(f"\nCalculating Bootstrap 95% CI for {name} (200 iterations)...")
    boot_mean, lower_ci, upper_ci = bootstrap_gini_ci(all_y_val, all_val_probs, n_iterations=200)

    results = {
        'Model': name,
        'Gini Index': boot_mean,
        'Gini 95% CI': f"[{lower_ci:.4f} - {upper_ci:.4f}]",
        'Recall @ Top 1%': np.mean(fold_recall_1),
        'PR-AUC': np.mean(fold_pr)
    }
    return results


def main():
    X, y, categorical_cols = load_data()

    model_factories = [
        (
            'LightGBM',
            lambda: lgb.LGBMClassifier(
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
        ),
        (
            'Random Forest',
            lambda: RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                class_weight='balanced',
                n_jobs=2,
                random_state=42,
                verbose=0
            )
        )
    ]

    if xgb_available:
        model_factories.append(
            (
                'XGBoost',
                lambda: XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=2
                )
            )
        )
    else:
        print('Warning: xgboost not installed; XGBoost baseline will be skipped.')

    baseline_results = []
    for name, factory in model_factories:
        res = run_cv_for_model(name, factory, X, y, categorical_cols)
        baseline_results.append(res)

    print('\n' + '=' * 85)
    print('FINAL BASELINE COMPARISON')
    print('=' * 85)

    summary_df = pd.DataFrame(baseline_results)

    # Comparison and result table
    columns_order = ['Model', 'Gini Index', 'Gini 95% CI', 'Recall @ Top 1%', 'PR-AUC']
    summary_df = summary_df[columns_order]

    print(summary_df.to_string(index=False, formatters={
        'Gini Index': '{:.6f}'.format,
        'Recall @ Top 1%': '{:.6f}'.format,
        'PR-AUC': '{:.5f}'.format
    }))

    # ============================================================================
    # SAVE BASELINE METRICS DYNAMICALLY FOR ABLATION STUDY
    # ============================================================================
    try:
        
        lgb_base_res = baseline_results[0]

        baseline_metrics = {
            'gini': float(lgb_base_res['Gini Index']),
            'pr_auc': float(lgb_base_res['PR-AUC']),
            'recall_1': float(lgb_base_res['Recall @ Top 1%'])
        }

        safe_output_dir = 'model_results'
        if not os.path.exists(safe_output_dir):
            os.makedirs(safe_output_dir)

        joblib.dump(baseline_metrics, os.path.join(safe_output_dir, 'baseline_metrics_backup.pkl'))
        print("\n✓ SUCCESS: Baseline metrics backup generated dynamically for Ablation Study.")

    except Exception as e:
        print(f"\nWarning: Could not save baseline metrics automatically ({e}).")


if __name__ == '__main__':
    main()