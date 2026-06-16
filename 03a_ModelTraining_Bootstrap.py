"""
PHASE 3a: Model Training, Cross-Validation & Bootstrap Analysis
================================================================
This script handles the computationally intensive parts:
- 5-fold Cross-Validation with LightGBM (Optimized Params)
- Probability Calibration with Isotonic Regression (Fold-Safe Split)
- 1000 Bootstrap iterations for Confidence Intervals
- Saves all results to disk to avoid memory overflow

Run AFTER 02c_HyperparameterTuning.py has generated preprocessed data files.
"""

import os
import gc
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc as sklearn_auc
from sklearn.calibration import IsotonicRegression
from sklearn.utils import resample
import lightgbm as lgb  

CONFIG = {
    "optuna_params_file":
        "model_results/best_lgbm_params.pkl"
}



warnings.filterwarnings('ignore')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def evaluate_ranking(y_true, y_probs, k_percent_list=[0.005, 0.01, 0.02]):
    """Calculate PR-AUC and Recall@K metrics."""
    results = {}
    
    # PR-AUC calculation
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    results['PR-AUC'] = sklearn_auc(recall, precision)
    
    # Recall@K calculation
    df_temp = pd.DataFrame({'y_true': y_true, 'y_probs': y_probs}).sort_values(by='y_probs', ascending=False)
    n_total_positives = y_true.sum()
    
    for k in k_percent_list:
        n_cutoff = max(1, int(len(y_true) * k))
        n_positives_at_k = df_temp.iloc[:n_cutoff]['y_true'].sum()
        recall_at_k = n_positives_at_k / n_total_positives if n_total_positives > 0 else 0
        results[f'Recall@{k*100}%'] = recall_at_k
        
    return results


def calculate_ece(y_true, y_probs, n_bins=10):
    """Calculate Expected Calibration Error."""
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)
    bins = np.linspace(0, 1, n_bins + 1)
    active_bin_indices = np.digitize(prob_pred, bins) - 1
    active_bin_indices = np.clip(active_bin_indices, 0, n_bins - 1)
    
    all_bin_indices = np.digitize(y_probs, bins) - 1
    all_bin_indices = np.clip(all_bin_indices, 0, n_bins - 1)
    bin_counts = np.bincount(all_bin_indices, minlength=n_bins)
    
    actual_weights = bin_counts[active_bin_indices] / len(y_true)
    ece = np.sum(np.abs(prob_true - prob_pred) * actual_weights)
    
    return ece


def plot_reliability_diagram(y_true, y_probs, n_bins=10, title="Reliability Diagram"):
    """Plots and saves an academic-grade Reliability Diagram (Calibration Curve)."""
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.plot(prob_pred, prob_true, marker='s', color='darkblue', linewidth=2, label='LightGBM (Calibrated)')
    
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Mean Predicted Probability', fontsize=10)
    plt.ylabel('Fraction of Positives (Actual)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    
    output_dir = 'model_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_png = os.path.join(output_dir, 'lgbm_reliability_diagram.png')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved reliability diagram to: {out_png}")


# ============================================================================
# LOAD PREPROCESSED DATA (from Phase 1)
# ============================================================================
print("Loading preprocessed data...")
X = pd.read_csv('X_preprocessed.csv').reset_index(drop=True)
y = pd.read_csv('y_preprocessed.csv').iloc[:, 0].reset_index(drop=True)
test_features = pd.read_csv('test_preprocessed.csv').reset_index(drop=True)
categorical_cols = joblib.load('categorical_cols.pkl')

# Training-derived rare class mapping (Leakage-free approach requested by advisor)
rare_map_path = 'rare_class_map.pkl'
if os.path.exists(rare_map_path):
    try:
        rare_class_map = joblib.load(rare_map_path)
        print(f"Loaded rare_class_map with {len(rare_class_map)} columns")
        for col, rare_vals in rare_class_map.items():
            if col in X.columns:
                X[col] = X[col].replace(rare_vals, -1)
            if col in test_features.columns:
                test_features[col] = test_features[col].replace(rare_vals, -1)
        print("Applied training-derived rare class mapping to train and test sets")
    except Exception as e:
        print(f"Warning: could not load/apply rare_class_map: {e}")
else:
    print("No rare_class_map found — ensure preprocessing saved it")

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Test features shape: {test_features.shape}\n")


# ============================================================================
# HYPERPARAMETER CONFIGURATION (Dynamic Optuna Loader)
# ============================================================================

print("=" * 70)
print("OPTUNA PARAMETERS USED IN TRAINING")
print("=" * 70)


lgb_params = joblib.load(
    CONFIG["optuna_params_file"]
)

for k, v in lgb_params.items():
    print(f"{k}: {v}")

print("=" * 70)

# ============================================================================
# PHASE 3.1: 5-FOLD CV WITH LEAKAGE-FREE CALIBRATION
# ============================================================================
print("=" * 70)
print("PHASE 3.1: 5-Fold Cross-Validation with Isotonic Calibration")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_calibrated = np.zeros(len(X))
test_preds_calibrated = np.zeros(len(test_features))
fold_models = []

ranking_metrics_list = []
brier_scores = []
ece_scores = []
fold_ginis = []

fold_ginis_raw_optimized = []
fold_pr_aucs_raw_optimized = []
fold_recalls_raw_optimized = []


current_cat_features = [c for c in categorical_cols if c in X.columns]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n--- Fold {fold}/5 ---")
    
    X_train_f = X.iloc[train_idx].copy()
    X_val_f = X.iloc[val_idx].copy()
    y_train_f = y.iloc[train_idx]
    y_val_f = y.iloc[val_idx]
    X_test_copy = test_features.copy()
    
    # METHODOLOGICAL REVISION: Fold-safe 80/20 split to avoid calibration leakage
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_f, y_train_f, test_size=0.2, stratify=y_train_f, random_state=42
    )
    
    # Safe category type casting (Prevents SettingWithCopyWarning)
    X_fit = X_fit.copy()
    X_cal = X_cal.copy()
    for col in current_cat_features:
        X_fit[col] = X_fit[col].astype('category')
        X_cal[col] = X_cal[col].astype('category')
        X_val_f[col] = X_val_f[col].astype('category')
        X_test_copy[col] = X_test_copy[col].astype('category')

    # MODEL TRAINING
    print("  Training LightGBM...")
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_cal, y_cal)],
        eval_metric="auc",
        categorical_feature=current_cat_features,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
    )
    
    # PREDICTIONS
    val_probs_raw = model.predict_proba(X_val_f)[:, 1]
    test_probs_raw = model.predict_proba(X_test_copy[X.columns])[:, 1]
    # Dynamically capture Step 2 (Optimized but Uncalibrated) metrics before Isotonic layer
    raw_auc = roc_auc_score(y_val_f, val_probs_raw)
    fold_ginis_raw_optimized.append(2 * raw_auc - 1)
    
    raw_ranking = evaluate_ranking(y_val_f, val_probs_raw)
    fold_pr_aucs_raw_optimized.append(raw_ranking['PR-AUC'])
    
    # Catch recall key safely
    r_key = [k for k in raw_ranking.keys() if 'Recall@1' in k][0]
    fold_recalls_raw_optimized.append(raw_ranking[r_key])




    cal_probs = model.predict_proba(X_cal)[:, 1]
    
    # CALIBRATION
    print("  Calibrating probabilities...")
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(cal_probs, y_cal)
    
    calibrated_fold_probs = iso_reg.transform(val_probs_raw)
    calibrated_test_probs = iso_reg.transform(test_probs_raw)
    
    # Store results
    oof_preds_calibrated[val_idx] = calibrated_fold_probs
    test_preds_calibrated += calibrated_test_probs / skf.n_splits
    fold_models.append({'model': model, 'iso_reg': iso_reg, 'fold': fold})
    
    # METRICS
    fold_gini = 2 * roc_auc_score(y_val_f, calibrated_fold_probs) - 1
    fold_brier = brier_score_loss(y_val_f, calibrated_fold_probs)
    fold_ece = calculate_ece(y_val_f, calibrated_fold_probs)
    metrics = evaluate_ranking(y_val_f, calibrated_fold_probs)
    
    fold_ginis.append(fold_gini)
    brier_scores.append(fold_brier)
    ece_scores.append(fold_ece)
    ranking_metrics_list.append(metrics)
    
    # PRINT LOGS: Clearly displaying both Step 2 (Raw) and Step 3 (Calibrated) for academic transparency
    print(f"  [Step 2 Raw Optimized] Gini: {2 * roc_auc_score(y_val_f, val_probs_raw) - 1:.5f} | PR-AUC: {raw_ranking['PR-AUC']:.5f}")
    print(f"  [Step 3 Calibrated   ] Gini: {fold_gini:.5f} | Brier: {fold_brier:.5f} | ECE: {fold_ece:.5f}")


    # MEMORY OPTIMIZATION 
    del X_train_f, X_val_f, X_test_copy, model, iso_reg
    del test_probs_raw, cal_probs, calibrated_fold_probs, calibrated_test_probs
    gc.collect()

# ============================================================================
# FINAL METRICS SUMMARY
# ============================================================================
final_gini = 2 * roc_auc_score(y, oof_preds_calibrated) - 1
final_brier = np.mean(brier_scores)
final_ece = np.mean(ece_scores)
final_pr_auc = np.mean([m['PR-AUC'] for m in ranking_metrics_list])

# Dynamic key selector to avoid Recall string alignment KeyError issues
first_metrics_dict = ranking_metrics_list[0]
recall_key = [k for k in first_metrics_dict.keys() if 'Recall@1' in k][0]
final_recall = np.mean([m[recall_key] for m in ranking_metrics_list])

print("\n" + "=" * 70)
print("FINAL CV RESULTS (After Calibration)")
print("=" * 70)
print(f"Overall Gini Index   : {final_gini:.5f}")
print(f"Mean Brier Score     : {final_brier:.5f}")
print(f"Mean ECE Score       : {final_ece:.5f}")
print(f"Mean PR-AUC          : {final_pr_auc:.5f}")
print(f"Mean Recall@1%       : {final_recall:.5f}")
print("=" * 70)


# ============================================================================
# PHASE 3.2: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
print("\n" + "=" * 70)
print("PHASE 3.2: Bootstrap Analysis (1000 iterations)")
print("=" * 70)

def calculate_bootstrap_ci(y_true, y_probs, n_bootstraps=1000):
    bootstrap_gini = []
    bootstrap_brier = []
    bootstrap_ece = []
    
    print(f"Running {n_bootstraps} bootstrap iterations...")
    
    for i in tqdm(range(n_bootstraps)):
        indices = resample(np.arange(len(y_true)), replace=True, random_state=i)
        
        y_true_sample = y_true.iloc[indices].values if hasattr(y_true, 'iloc') else y_true[indices]
        y_probs_sample = y_probs[indices]
        
        sample_auc = roc_auc_score(y_true_sample, y_probs_sample)
        bootstrap_gini.append(2 * sample_auc - 1)
        bootstrap_brier.append(brier_score_loss(y_true_sample, y_probs_sample))
        bootstrap_ece.append(calculate_ece(y_true_sample, y_probs_sample))
    
    metrics = {
        'Gini Index': bootstrap_gini,
        'Brier Score': bootstrap_brier,
        'ECE': bootstrap_ece
    }
    
    ci_results = {}
    for name, values in metrics.items():
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        mean = np.mean(values)
        ci_results[name] = (lower, mean, upper)
        
    return ci_results

ci_results = calculate_bootstrap_ci(y, oof_preds_calibrated, n_bootstraps=1000)

print("\n" + "=" * 70)
print("95% CONFIDENCE INTERVALS (Bootstrap)")
print("-" * 70)
for metric, values in ci_results.items():
    print(f"{metric:<12}: {values[1]:.5f} (95% CI: [{values[0]:.5f}, {values[2]:.5f}])")
print("=" * 70)


# ============================================================================
# SAVE RESULTS TO DISK
# ============================================================================
print("\nSaving results to disk...")
output_dir = 'model_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.save(f'{output_dir}/oof_preds_calibrated.npy', oof_preds_calibrated)
np.save(f'{output_dir}/test_preds_calibrated.npy', test_preds_calibrated)
joblib.dump(fold_models, f'{output_dir}/fold_models.pkl')

metrics_summary = {
    'final_gini': final_gini,
    'final_brier': final_brier,
    'final_ece': final_ece,
    'final_pr_auc': final_pr_auc,
    'final_recall': final_recall,
    'fold_ginis': fold_ginis,
    'brier_scores': brier_scores,
    'ece_scores': ece_scores,
    'ranking_metrics_list': ranking_metrics_list,
    'ci_results': ci_results
}
joblib.dump(metrics_summary, f'{output_dir}/metrics_summary.pkl')

fold_results_df = pd.DataFrame({
    'Fold': list(range(1, 6)),
    'Gini': fold_ginis,
    'Brier': brier_scores,
    'ECE': ece_scores,
    'PR-AUC': [m['PR-AUC'] for m in ranking_metrics_list],
    'Recall@1%': [m[recall_key] for m in ranking_metrics_list]
})
fold_results_df.to_csv(f'{output_dir}/fold_results.csv', index=False)


# ============================================================================
# FINAL ABLATION STUDY & MODEL EVOLUTION REPORT (100% AUTOMATED & ZERO CONSTANTS)
# ============================================================================
print("\n" + "=" * 90)
print("FINAL ABLATION STUDY & MODEL EVOLUTION (LEAKAGE-FREE & HONEST PERFORMANCE)")
print("=" * 90)

# --- DYNAMIC STEP 1: Strict loading from Baseline Backup file ---
baseline_file = f'{output_dir}/baseline_metrics_backup.pkl'
if not os.path.exists(baseline_file):
    raise FileNotFoundError(
        "CRITICAL ERROR: 'baseline_metrics_backup.pkl' not found! "
        "Please run your baseline execution script first to log Step 1 metrics."
    )

try:
    base_backup = joblib.load(baseline_file)
    base_gini = float(base_backup['gini'])
    base_pr = float(base_backup['pr_auc'])
    base_recall = float(base_backup['recall_1'])
    print("✓ SUCCESS: Baseline metrics loaded dynamically from disk.")
except Exception as e:
    raise RuntimeError(f"CRITICAL ERROR: Failed to parse Baseline backup file. Details: {e}")

# --- DYNAMIC STEP 2: Computed LIVE from the uncalibrated raw predictions generated inside this run ---
optuna_gini = np.mean(fold_ginis_raw_optimized)
optuna_pr = np.mean(fold_pr_aucs_raw_optimized)
optuna_recall = np.mean(fold_recalls_raw_optimized)

# --- DYNAMIC STEP 3: Generated live from calibrated current script run execution data ---
ablation_data = [
    {
        'Experiment Step': '1. LightGBM Baseline (Default Params)', 
        'Gini Index': base_gini, 
        'PR-AUC': base_pr, 
        'Recall@1%': base_recall, 
        'Improvement (Δ)': 0.00000
    },
    {
        'Experiment Step': '2. LightGBM Optimized (Optuna Tuning)', 
        'Gini Index': optuna_gini, 
        'PR-AUC': optuna_pr, 
        'Recall@1%': optuna_recall, 
        'Improvement (Δ)': optuna_gini - base_gini
    },
    {
        'Experiment Step': '3. LightGBM Final (Tuned + Calibration)', 
        'Gini Index': final_gini, 
        'PR-AUC': final_pr_auc, 
        'Recall@1%': final_recall, 
        'Improvement (Δ)': final_gini - base_gini
    }
]

ablation_df = pd.DataFrame(ablation_data)
print(ablation_df.to_string(index=False, formatters={'Gini Index': '{:.5f}'.format, 'PR-AUC': '{:.5f}'.format, 'Recall@1%': '{:.5f}'.format, 'Improvement (Δ)': '{:+.5f}'.format}))
print("=" * 90)
ablation_df.to_csv(f'{output_dir}/ablation_study.csv', index=False)
print(f"✓ Saved fully dynamic ablation study summary to: {output_dir}/ablation_study.csv")



# ============================================================================
# CREATE KAGGLE SUBMISSION
# ============================================================================
print("\nCreating Kaggle submission...")
test_ids = pd.read_csv('test.csv')['id']
submission_df = pd.DataFrame({'id': test_ids, 'target': test_preds_calibrated})
submission_df.to_csv(f'{output_dir}/submission_final.csv', index=False)
print(f"✓ Submission saved to '{output_dir}/submission_final.csv'")


# ============================================================================
# VISUALIZE CALIBRATION RESULTS
# ============================================================================
print("\nTraining final production model on full dataset...")

final_model = lgb.LGBMClassifier(**lgb_params)

for col in current_cat_features:
    X[col] = X[col].astype("category")

final_model.fit(
    X,
    y,
    categorical_feature=current_cat_features
)
# ============================================================================
# SAVE FINAL MODEL
# ============================================================================
print("\nSAVING FINAL MODEL")
print("Best iteration:", final_model.best_iteration_)

if hasattr(final_model, "booster_"):
    print("Num trees:", final_model.booster_.num_trees())

joblib.dump(
    final_model,
    "model_results/final_model.pkl"
)

print("✓ Saved final_model.pkl")


print("\nGenerating calibration plots...")
plot_reliability_diagram(y, oof_preds_calibrated, title="Reliability Diagram (After Isotonic Calibration)")

print("\n" + "=" * 70)
print("PHASE 03a COMPLETED SUCCESSFULLY")
print("=" * 70)

