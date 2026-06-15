"""
PHASE 2 BASELINE: Model comparison for selection (Top 20 Features & Bootstrap CI)
================================================================================
This script compares LightGBM, RandomForest, and XGBoost using the
top 20 most important features. It computes a 95% Confidence Interval for
the Gini index using Bootstrap resampling on the out-of-fold predictions.

Run after 01_Preprocessing.py has produced preprocessed data and
before 02_ModelTraining_Bootstrap.py to confirm the best model.
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

# The most important 20 features from previous step
TOP_20_FEATURES = [
    'ps_car_11_cat', 'ps_ind_03', 'ps_car_13', 'ps_car_01_cat', 'ps_car_06_cat',
    'ps_ind_15', 'ps_reg_03', 'ps_reg_01', 'ps_ind_01', 'ps_ind_05_cat',
    'ps_car_15', 'ps_reg_02', 'ps_car_09_cat', 'ps_car_14', 'ps_ind_02_cat',
    'ps_ind_17_bin', 'ps_ind_04_cat', 'ps_car_07_cat', 'ps_car_04_cat', 'ps_car_12'
]


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


def get_importances_from_model(model, feature_names):
    """Return a pandas Series of importances aligned to feature_names or raise."""
    try:
        booster = getattr(model, 'booster_', None) or getattr(model, 'booster', None)
        if booster is not None:
            gains = booster.feature_importance(importance_type='gain')
            if gains is not None and len(gains) == len(feature_names):
                return pd.Series(gains, index=feature_names)
    except Exception:
        pass

    try:
        imp = getattr(model, 'feature_importances_', None)
        if imp is not None and len(imp) == len(feature_names):
            return pd.Series(imp, index=feature_names)
    except Exception:
        pass

    raise RuntimeError('Could not extract feature importances from model')


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

    # ===== TOP 20 FEATURE =====
    existing_top_features = [c for c in TOP_20_FEATURES if c in X.columns]
    X = X[existing_top_features].copy()
    print(f'Filtered to Top 20 features. X shape: {X.shape}, y shape: {y.shape}')
    
    return X, y, categorical_cols


def run_cv_for_model(name, model_factory, X, y, categorical_cols, n_splits=5):
    print('\n' + '=' * 70)
    print(f'CV STARTING FOR: {name}')
    print('=' * 70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Tüm katlamaların validasyon tahminlerini Bootstrap için havuzda biriktiriyoruz
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
    print('FINAL BASELINE COMPARISON (TOP 20 FEATURES & BOOTSTRAP GINI)')
    print('=' * 85)
    
    summary_df = pd.DataFrame(baseline_results)
    
    # Tam olarak hedeflediğiniz çıktı tablosu sütun sırası
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
    # Get the first model results (LightGBM Baseline) from the list
    try:
        # baseline_results listesindeki ilk satırı (LightGBM Baseline) çekiyoruz
        # Hata​​sız dinamik eşleme için sütun adları tablonuzla birebir eşitlendi
        lgb_base_res = baseline_results[0]
        
        baseline_metrics = {
            'gini': float(lgb_base_res['Gini Index']),
            'pr_auc': float(lgb_base_res['PR-AUC']),
            'recall_1': float(lgb_base_res['Recall @ Top 1%'])
        }
        
        # Eliminate any global NameError issues using an explicit directory string
        safe_output_dir = 'model_results'
        if not os.path.exists(safe_output_dir):
            os.makedirs(safe_output_dir)
            
        joblib.dump(baseline_metrics, os.path.join(safe_output_dir, 'baseline_metrics_backup.pkl'))
        print("\n✓ SUCCESS: Baseline metrics backup generated dynamically for Ablation Study.")
        
    except Exception as e:
        print(f"\nWarning: Could not save baseline metrics automatically ({e}).")



if __name__ == '__main__':
    main()


"""
import os
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sklearn_auc
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Bir önceki adımda elde edilen en önemli 20 değişken (KOD2 Setup - 5-Fold Averaged)
TOP_20_FEATURES = [
    'ps_car_11_cat', 'ps_ind_03', 'ps_car_13', 'ps_car_01_cat', 'ps_car_06_cat',
    'ps_ind_15', 'ps_reg_03', 'ps_reg_01', 'ps_ind_01', 'ps_ind_05_cat',
    'ps_car_15', 'ps_reg_02', 'ps_car_09_cat', 'ps_car_14', 'ps_ind_02_cat',
    'ps_ind_17_bin', 'ps_ind_04_cat', 'ps_car_07_cat', 'ps_car_04_cat', 'ps_car_12'
]


def evaluate_ranking(y_true, y_probs, k_percent_list=[0.005, 0.01, 0.02]):
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


def get_importances_from_model(model, feature_names):
    #Return a pandas Series of importances aligned to feature_names or raise.
    try:
        booster = getattr(model, 'booster_', None) or getattr(model, 'booster', None)
        if booster is not None:
            gains = booster.feature_importance(importance_type='gain')
            if gains is not None and len(gains) == len(feature_names):
                return pd.Series(gains, index=feature_names)
    except Exception:
        pass

    try:
        imp = getattr(model, 'feature_importances_', None)
        if imp is not None and len(imp) == len(feature_names):
            return pd.Series(imp, index=feature_names)
    except Exception:
        pass

    raise RuntimeError('Could not extract feature importances from model')


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

    # ===== TOP 20 FEATURE =====
    existing_top_features = [c for c in TOP_20_FEATURES if c in X.columns]
    X = X[existing_top_features].copy()
    print(f'Filtered to Top 20 features. X shape: {X.shape}, y shape: {y.shape}')
    
    return X, y, categorical_cols


def run_cv_for_model(name, model_factory, X, y, categorical_cols, n_splits=5):
    print('\n' + '=' * 70)
    print(f'CV STARTING FOR: {name}')
    print('=' * 70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_ginis = []
    fold_pr = []
    fold_recall_1 = []
    fold_imps = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Geçerli kategorik kolonları (sadece Top 20 içinde olanlar) bulma
        current_cat_features = [c for c in categorical_cols if c in X_train.columns]

        if name == 'LightGBM':
            # KOD2 Setup: ps_car_11_cat dahil tüm kategorikler ham bırakılır, LightGBM'e bildirilir
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
            # RF ve XGBoost için: ps_car_11_cat seçilen top 20 içindeyse sızıntısız Target Encoding uygulanır
            if 'ps_car_11_cat' in X_train.columns:
                enc_map, global_mean = fit_target_encoding_map(X_train['ps_car_11_cat'], y_train, smoothing=10)
                X_train['ps_car_11_cat'] = apply_target_encoding(X_train['ps_car_11_cat'], enc_map, global_mean)
                X_val['ps_car_11_cat'] = apply_target_encoding(X_val['ps_car_11_cat'], enc_map, global_mean)
            
            # Diğer kategorikler label encoding (cat.codes) yapılır
            other_cats = [c for c in current_cat_features if c != 'ps_car_11_cat']
            for col in other_cats:
                X_train[col] = X_train[col].astype('category').cat.codes
                X_val[col] = X_val[col].astype('category').cat.codes

            model = model_factory()
            model.fit(X_train, y_train)

        try:
            imp_series = get_importances_from_model(model, X_train.columns.tolist())
            fold_imps.append(imp_series)
        except Exception:
            pass

        val_probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        gini = 2 * auc - 1
        ranking = evaluate_ranking(y_val, val_probs)

        fold_ginis.append(gini)
        fold_pr.append(ranking['PR-AUC'])
        fold_recall_1.append(ranking['Recall@1%'])

        print(f'Fold {fold} | Gini: {gini:.5f} | PR-AUC: {ranking["PR-AUC"]:.5f} | Recall@1%: {ranking["Recall@1%"]:.5f}')

    results = {
        'Model': name,
        'Mean Gini': np.mean(fold_ginis),
        'Std Gini': np.std(fold_ginis),
        'Mean PR-AUC': np.mean(fold_pr),
        'Std PR-AUC': np.std(fold_pr),
        'Mean Recall@1%': np.mean(fold_recall_1),
        'Std Recall@1%': np.std(fold_recall_1),
        'Fold Ginis': fold_ginis,
        'Fold PR-AUCs': fold_pr,
        'Fold Recall@1%': fold_recall_1,
    }

    if len(fold_imps) > 0:
        try:
            imps_df = pd.concat(fold_imps, axis=1)
            results['Importances'] = imps_df.mean(axis=1)
        except Exception:
            results['Importances'] = None
    else:
        results['Importances'] = None
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

    print('\n' + '=' * 70)
    print('FINAL BASELINE COMPARISON (TOP 20 FEATURES)')
    print('=' * 70)
    
    summary_data = []
    for r in baseline_results:
        summary_data.append({
            'Model': r['Model'],
            'Mean Gini': f"{r['Mean Gini']:.5f} (±{r['Std Gini']:.5f})",
            'Mean PR-AUC': f"{r['Mean PR-AUC']:.5f}",
            'Mean Recall@1%': f"{r['Mean Recall@1%']:.5f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))


   

if __name__ == '__main__':
    main()

"""