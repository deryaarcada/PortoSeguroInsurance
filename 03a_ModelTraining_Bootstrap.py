"""
PHASE 3a: Model Training, Cross-Validation & Bootstrap Analysis
================================================================
This script handles the computationally intensive parts:
- 5-fold Cross-Validation with LightGBM (Optimized Params)
- Probability Calibration with Isotonic Regression (Fold-Safe Split)
- 1000 Bootstrap iterations for Confidence Intervals
- Saves all results to disk 

Run AFTER 02b_HyperparameterTuning.py has generated preprocessed data files.
"""

import os
import gc
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import IsotonicRegression
import lightgbm as lgb

from metrics import calculate_ece, evaluate_ranking, calculate_bootstrap_ci
from preprocessing_utils import fit_rare_map, apply_rare_map
 
CONFIG = {
    "optuna_params_file":
        "model_results/best_lgbm_params.pkl"
}
 
 
warnings.filterwarnings('ignore')
 
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
#
# NOTE: evaluate_ranking() and calculate_ece() previously lived here as local
# copies. They now live in metrics.py, the single stsndart source for these
# functions. calculate_ece() in particular used to mix quantile bins with equal-width bin weights, 
# which silently distorted the ECE estimate; that bug is fixed in metrics.py.
# ============================================================================

def predict_calibrated_ensemble(bundle, X_new):
    """
    Score new data with the SAME predictor used for validation and the
    Kaggle submission: the 5-fold LightGBM ensemble, each fold's raw score
    passed through that fold's own isotonic calibrator, then averaged.

    This is the counterpart to `model_results/calibrated_predictor.pkl`
    (see the "SAVE THE REAL, CALIBRATED PREDICTOR" section below for why a
    dedicated bundle exists). Loading `final_model.pkl` and calling
    `.predict_proba()` on it does NOT reproduce these numbers — that file
    is a single uncalibrated LightGBM model kept for interpretability
    (SHAP) purposes only.

    NOTE (fold-safe preprocessing fix): each fold's rare-category mapping is
    now learned from only that fold's own training rows (see
    preprocessing_utils.py), so it is no longer a single bundle-wide
    mapping — it is stored per fold, inside each `fold_models` entry
    ('rare_map' key), and applied with that fold's own model before scoring.

    Parameters
    ----------
    bundle : dict
        A bundle as saved to `calibrated_predictor.pkl`, with keys
        'fold_models' (list of {'model', 'iso_reg', 'rare_map', 'fold'}),
        'feature_names', 'categorical_features'.
    X_new : pd.DataFrame
        Raw (pre-rare-mapping) feature frame with the same columns as
        training data, in any column order.

    Returns
    -------
    np.ndarray
        Calibrated probability of the positive class, one per row of X_new,
        averaged across all fold models.
    """
    preds = np.zeros(len(X_new))
    for fm in bundle['fold_models']:
        X_scored = apply_rare_map(X_new, fm['rare_map'])
        X_scored = X_scored[bundle['feature_names']]
        for col in bundle['categorical_features']:
            X_scored[col] = X_scored[col].astype('category')

        raw_probs = fm['model'].predict_proba(X_scored)[:, 1]
        preds += fm['iso_reg'].transform(raw_probs) / len(bundle['fold_models'])

    return preds


def plot_reliability_diagram(y_true, y_probs, n_bins=10, title="Reliability Diagram"):
    """Plots and saves an academic-grade Reliability Diagram (Calibration Curve)."""
    from sklearn.calibration import calibration_curve
 
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='quantile')
 
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

    # ============================================================
    # ZOOMED RELIABILITY DIAGRAM (0-10%)
    # ============================================================

    plt.figure(figsize=(6, 6))

    plt.plot(
        [0, 0.10],
        [0, 0.10],
        linestyle='--',
        color='gray',
        label='Perfect Calibration'
    )

    plt.plot(
        prob_pred,
        prob_true,
        marker='s',
        color='darkblue',
        linewidth=2,
        label='LightGBM (Calibrated)'
    )

    plt.xlim(0, 0.10)
    plt.ylim(0, 0.10)

    plt.title(
        'Reliability Diagram (Zoomed: 0–10%)',
        fontsize=12,
        fontweight='bold',
        pad=12
    )

    plt.xlabel(
        'Mean Predicted Probability',
        fontsize=10
    )

    plt.ylabel(
        'Fraction of Positives (Actual)',
        fontsize=10
    )

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    out_png_zoomed = os.path.join(
        output_dir,
        'lgbm_reliability_diagram_zoomed.png'
    )

    plt.savefig(
        out_png_zoomed,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()
    print(f"✓ Zoomed reliability diagram saved to: {out_png_zoomed}")

 
 
# ============================================================================
# LOAD PREPROCESSED DATA (from Phase 1)
# ============================================================================
print("Loading preprocessed data...")
X = pd.read_csv('X_preprocessed.csv').reset_index(drop=True)
y = pd.read_csv('y_preprocessed.csv').iloc[:, 0].reset_index(drop=True)
test_features = pd.read_csv('test_preprocessed.csv').reset_index(drop=True)
categorical_cols = joblib.load('categorical_cols.pkl')
cols_for_rare = joblib.load('cols_for_rare.pkl')

# NOTE The mapping is fit fresh, per fold, on only that fold's training partition inside
# the CV loop below (see preprocessing_utils.py), and separately on the
# full training set for the final production model (legitimate there, since
# that model is never evaluated on held-out data). X and test_features are
# therefore intentionally left un-mapped here.
print(f"Loaded {len(cols_for_rare)} candidate columns for rare-category mapping "
      f"(thresholding deferred to each CV fold)")
 
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Test features shape: {test_features.shape}\n")
 
 
# ============================================================================
# HYPERPARAMETER CONFIGURATION (Dynamic Optuna Loader)
# ============================================================================
 
print("=" * 70)
print("OPTUNA PARAMETERS USED IN TRAINING")
print("=" * 70)
 
lgb_params_raw = joblib.load(
    CONFIG["optuna_params_file"]
)
 

lgb_params = dict(lgb_params_raw)
n_estimators_effective = lgb_params.pop('n_estimators_effective', None)
 
for k, v in lgb_params.items():
    print(f"{k}: {v}")
if n_estimators_effective is not None:
    print(f"n_estimators_effective (reserved for final model only): {n_estimators_effective}")
 
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

    # FOLD-SAFE rare-category mapping: fit only on this fold's outer
    # training rows (X_train_f), then apply to this fold's validation rows
    # AND the test set, so the test-set scoring stays consistent with what
    # this fold's model was actually trained on. The 80/20 fit/calibration
    # split below happens entirely within the already-mapped X_train_f, so
    # it doesn't need (or get) its own separate rare map.
    fold_rare_map = fit_rare_map(X_train_f, cols_for_rare, threshold=0.01)
    X_train_f = apply_rare_map(X_train_f, fold_rare_map)
    X_val_f = apply_rare_map(X_val_f, fold_rare_map)
    X_test_copy = apply_rare_map(X_test_copy, fold_rare_map)
 
    # Fold-safe 80/20 split to avoid calibration leakage
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_f, y_train_f, test_size=0.2, stratify=y_train_f, random_state=42
    )
 
    # Safe category type casting
    X_cal = X_cal.copy()
    for col in current_cat_features:
        X_fit[col] = X_fit[col].astype('category')
        X_cal[col] = X_cal[col].astype('category')
        X_val_f[col] = X_val_f[col].astype('category')
        X_test_copy[col] = X_test_copy[col].astype('category')
 
    # MODEL TRAINING
    # Uses the ORIGINAL n_estimators (upper bound from Optuna's search space)
    # together with early stopping, exactly as during tuning, this lets each
    # fold find its own optimal tree count, consistent with how Optuna scored it.
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
    fold_recalls_raw_optimized.append(raw_ranking['Recall@1%'])
 
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
    fold_models.append({'model': model, 'iso_reg': iso_reg, 'rare_map': fold_rare_map, 'fold': fold})
 
    # METRICS
    fold_gini = 2 * roc_auc_score(y_val_f, calibrated_fold_probs) - 1
    fold_brier = brier_score_loss(y_val_f, calibrated_fold_probs)
    fold_ece = calculate_ece(y_val_f, calibrated_fold_probs)
    metrics = evaluate_ranking(y_val_f, calibrated_fold_probs)
 
    fold_ginis.append(fold_gini)
    brier_scores.append(fold_brier)
    ece_scores.append(fold_ece)
    ranking_metrics_list.append(metrics)
 
    # PRINT LOGS: Clearly displaying both Step 2 (Raw) and Step 3 (Calibrated)
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

recall_key = 'Recall@1%'
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
 
# calculate_bootstrap_ci() is now imported from metrics.py (same canonical
# module used for calculate_ece / evaluate_ranking). Its behavior is
# unchanged from the local version this replaces: 1000 bootstrap resamples,
# reporting (lower, mean, upper) 95% CI for Gini, Brier, and ECE together.
print(f"Running 1000 bootstrap iterations...")
ci_results = calculate_bootstrap_ci(y, oof_preds_calibrated, n_bootstraps=1000, show_progress=True)
 
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

# ============================================================================
# SAVE THE REAL, CALIBRATED PREDICTOR
# ============================================================================
# `fold_models.pkl` holds everything needed to reproduce the exact
# probabilities used for OOF validation and the Kaggle submission: each
# fold's LightGBM model, that fold's isotonic calibrator, and that fold's
# own rare-category map. `model_results/final_model.pkl` (saved further
# below) is a different, uncalibrated model kept only for SHAP — its
# `.predict_proba()` does NOT give the same calibrated probability.
#
# To make the actual scoring predictor an explicit, self-describing
# artifact, this bundles the fold ensemble together with everything
# needed to preprocess raw input the same way (feature order, which
# columns are categorical). Each fold's own rare-category map travels
# WITH that fold inside `fold_models`, rather than as one shared,
# bundle-wide mapping. Score new data with
# `predict_calibrated_ensemble(bundle, X_new)` (defined above).
model_bundle = {
    "kind": "calibrated_fold_ensemble",
    "fold_models": fold_models,
    "feature_names": list(X.columns),
    "categorical_features": current_cat_features,
}
joblib.dump(model_bundle, f'{output_dir}/calibrated_predictor.pkl')
print(f"✓ Saved calibrated_predictor.pkl — the actual predictor behind "
      f"OOF validation and submission_final.csv")

# Sanity check: scoring the raw test set through the bundle should reproduce
# test_preds_calibrated.npy exactly (both are the same fold-averaged,
# isotonic-calibrated computation, each using that fold's own rare map),
# which confirms the bundle is faithful.
_bundle_test_preds = predict_calibrated_ensemble(model_bundle, test_features)
_max_diff = np.max(np.abs(_bundle_test_preds - test_preds_calibrated))
print(f"  Sanity check — max |bundle prediction - test_preds_calibrated.npy|: {_max_diff:.2e}")
if _max_diff > 1e-8:
    print("  Warning: bundle predictions do not exactly match test_preds_calibrated.npy — investigate.")

 
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
# FINAL ABLATION STUDY & MODEL EVOLUTION REPORT
# ============================================================================
print("\n" + "=" * 90)
print("FINAL ABLATION STUDY & MODEL EVOLUTION")
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
# TRAIN + SAVE FINAL PRODUCTION MODEL (full dataset, no early stopping)
# ============================================================================
print("\nTraining final production model on full dataset...")
 
# ----------------------------------------------------------------------
# FIX: There is no held-out validation set here, so early stopping cannot
# be used to find the tree count. Instead we directly set n_estimators to
# 'n_estimators_effective' — the average early-stopped tree count Optuna
# actually used for its best trial — so this final model's capacity
# matches the tuning result it is supposed to represent, rather than
# silently training with the (larger) Optuna search-space upper bound.
# ----------------------------------------------------------------------
final_params = dict(lgb_params)
if n_estimators_effective is not None:
    final_params['n_estimators'] = int(round(n_estimators_effective))
    print(f"Using n_estimators={final_params['n_estimators']} "
          f"(Optuna's early-stopped average) for the final model.")
else:
    print("Warning: 'n_estimators_effective' not found in saved params — "
          f"falling back to the Optuna upper bound n_estimators={final_params.get('n_estimators')}.")
 
final_model = lgb.LGBMClassifier(**final_params)

final_rare_map = fit_rare_map(X, cols_for_rare, threshold=0.01)
X = apply_rare_map(X, final_rare_map)

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
# IMPORTANT: this is a single LightGBM model trained on ALL data, with NO
# isotonic calibration layer. Its .predict_proba() output is NOT the same
# probability used for OOF validation or submission_final.csv — those come
# from the 5-fold calibrated ensemble saved just above as
# `calibrated_predictor.pkl`. This raw model is kept specifically because
# 03b_SHAP_Interpretability.py needs a plain tree model for TreeSHAP (SHAP's
# TreeExplainer does not work through an isotonic regression step).
#
# Rule of thumb: use `calibrated_predictor.pkl` for scoring, validation
# reproduction, audits, or a demo. Use `final_model.pkl` only for SHAP /
# feature-importance style interpretability work.
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
 