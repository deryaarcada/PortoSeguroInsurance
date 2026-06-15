"""
PHASE 3B: Model Interpretability with SHAP
================================================
This script handles SHAP calculations which are computationally expensive.
Run AFTER Phase 2 has generated model results.

NOTE: SHAP calculations should run on a sample of data to avoid memory overflow.
For full dataset, consider using SHAP on a representative sample.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc
import warnings
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

warnings.filterwarnings('ignore')


# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================

print("=" * 70)
print("PHASE 3: SHAP Interpretability Analysis")
print("=" * 70)

print("\nLoading preprocessed data and trained models...")

X = pd.read_csv('X_preprocessed.csv')
y = pd.read_csv('y_preprocessed.csv').iloc[:, 0]

# Utility: target-encoding helpers (fit on TRAIN only)
def fit_target_encoding_map(cat_train, y_train, smoothing=10):
    global_mean = y_train.mean()
    agg = y_train.groupby(cat_train).agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    smooth_map = (counts * means + smoothing * global_mean) / (counts + smoothing)
    return smooth_map.to_dict(), global_mean

def apply_target_encoding(cat_series, enc_map, global_mean):
    return cat_series.map(enc_map).fillna(global_mean)

# Try to load an existing final model; if missing, train a final model on full training data
final_model_path = 'model_results/final_model.pkl'

print('Checking for final model...')
final_model = None
if os.path.exists(final_model_path):
    final_model = joblib.load(final_model_path)
    print('Loaded existing final model.')
else:
    print('No final model found — will train a final model using full training data')



print("=" * 70)
print("MODEL VERIFICATION")
print("=" * 70)

print("Model type:", type(final_model))

if hasattr(final_model, "booster_"):
    print("Best iteration:", final_model.best_iteration_)
    print("Num trees:", final_model.booster_.num_trees())

print("Feature count in model:", final_model.n_features_)
print("Feature count in X:", X.shape[1])

print("=" * 70)




# Determine categorical columns
try:
    categorical_cols = joblib.load('categorical_cols.pkl')
except Exception:
    categorical_cols = []

# If final model not present, compute full-train TE for ps_car_11_cat and train final model

""""
if final_model is None:
    if 'ps_car_11_cat' in X.columns:
        print('Fitting full-train target encoding for ps_car_11_cat...')
        enc_map, global_mean = fit_target_encoding_map(X['ps_car_11_cat'], y, smoothing=10)
        X['ps_car_11_cat'] = apply_target_encoding(X['ps_car_11_cat'], enc_map, global_mean)
    else:
        print('ps_car_11_cat not present in X; skipping TE')

    lgb_params = {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "num_leaves": 7,
        "max_depth": 7,
        "min_child_samples": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": 42, 
        "n_jobs": 2,
        "verbosity": -1
    }

    current_cat_features = [c for c in categorical_cols if c != 'ps_car_11_cat']
    for col in current_cat_features:
        if col in X.columns:
            X[col] = X[col].astype('category')

    print(f"Training final LightGBM on full data (n_estimators={lgb_params['n_estimators']})...")
    final_model = lgb.LGBMClassifier(**lgb_params)
    # use a small holdout for early stopping to avoid overfitting long runs
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.05, stratify=y, random_state=42)
    final_model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        categorical_feature=current_cat_features,
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)]
    )

    joblib.dump(final_model, final_model_path)
    joblib.dump({'enc_map': enc_map, 'global_mean': global_mean}, 'model_results/te_ps_car_11_cat_fulltrain.pkl')
    print('Saved final model and TE map.')

else:
    # If final model exists, ensure we still apply TE to X for SHAP input if TE map exists
    te_path = 'model_results/te_ps_car_11_cat_fulltrain.pkl'
    if os.path.exists(te_path) and 'ps_car_11_cat' in X.columns:
        te = joblib.load(te_path)
        X['ps_car_11_cat'] = apply_target_encoding(X['ps_car_11_cat'], te['enc_map'], te['global_mean'])
"""
final_model_path = "model_results/final_model.pkl"

if not os.path.exists(final_model_path):
    raise FileNotFoundError(
        "final_model.pkl not found. "
        "Run 03a_ModelTraining_Bootstrap.py first."
    )

final_model = joblib.load(final_model_path)

print("✓ Loaded final production model.")


print(f"X shape: {X.shape}\n")

# ============================================================================
# SHAP ANALYSIS ON SAMPLE (Memory-Efficient)
# ============================================================================

print("Preparing sample data for SHAP analysis (1000 random samples)...")

# Use the final full-train model for SHAP
model = final_model

# Sample data - IMPORTANT: SHAP is expensive, use sample instead of full dataset
sample_size = min(1000, len(X))
np.random.seed(42)
sample_indices = np.random.choice(len(X), size=sample_size, replace=False)

X_sample = X.iloc[sample_indices].copy()
y_sample = y.iloc[sample_indices].values

# Ensure categorical columns are properly typed
categorical_cols = joblib.load('categorical_cols.pkl')
current_cat_features = [c for c in categorical_cols if c != 'ps_car_11_cat']

for col in current_cat_features:
    X_sample[col] = X_sample[col].astype('category')

print(f"Sample size for SHAP: {X_sample.shape[0]} observations")

# ============================================================================
# CALCULATE SHAP VALUES
# ============================================================================

print("\nCalculating SHAP values...")
print("(This may take 2-5 minutes depending on your hardware)\n")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# For binary classification, shap_values is a list of 2 arrays (one per class)
# Use the positive class (index 1)
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
else:
    shap_values_pos = shap_values

print("[OK] SHAP values calculated\n")

# ============================================================================
# SHAP VISUALIZATIONS
# ============================================================================

output_dir = 'model_results/shap_analysis'
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ========== SHAP Summary Plot (Dot) ==========
print("Creating SHAP Summary Plot (Dot)...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_pos,
    X_sample,
    plot_type="dot",
    show=False,
    max_display=X_sample.shape[1]
)
plt.title("SHAP Summary Plot: Feature Impact on Model Output", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_shap_summary_dot.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved to: 01_shap_summary_dot.png")

# ========== SHAP Summary Plot (Bar) ==========
print("Creating SHAP Summary Plot (Bar)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values_pos,
    X_sample,
    plot_type="bar",
    show=False,
    max_display=X_sample.shape[1]
)
plt.title("SHAP Mean Absolute Impact on Model Output", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved to: 02_shap_summary_bar.png")

# ========== SHAP Feature Importance Table ==========
print("Creating Feature Importance Table...")

# Calculate mean absolute SHAP values per feature
mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Feature': X_sample.columns,
    'Mean |SHAP|': mean_abs_shap,
    'Mean SHAP': shap_values_pos.mean(axis=0)
}).sort_values('Mean |SHAP|', ascending=False)

print("\nTop 15 Most Important Features (by SHAP):")
print(shap_importance_df.head(15).to_string(index=False))

# Save importance table
shap_importance_df.to_csv(f'{output_dir}/shap_feature_importance.csv', index=False)
print(f"[OK] Saved to: shap_feature_importance.csv")

# ========== Top 20 Features Bar Plot ==========
print("\nCreating Top 20 Feature Importance Bar Plot...")
top_20_df = shap_importance_df.head(20)
plt.figure(figsize=(10, 8))
plt.barh(range(len(top_20_df)), top_20_df['Mean |SHAP|'].values[::-1])
plt.yticks(range(len(top_20_df)), top_20_df['Feature'].values[::-1])
plt.xlabel('Mean |SHAP value|', fontsize=12)
plt.title('Top 20 Features by SHAP Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02b_shap_top20_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved to: 02b_shap_top20_bar.png")

# ========== SHAP Dependence Plots (Top 6 Features) ==========
print("\nCreating SHAP Dependence Plots for top 6 features...")

top_6_features = shap_importance_df.head(6)['Feature'].values

for idx, feature in enumerate(top_6_features, 1):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature, shap_values_pos, X_sample, 
        feature_names=X_sample.columns,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_shap_dependence_{idx:02d}_{feature}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {idx}/6: {feature}")

# ========== SHAP Force Plot (Individual Predictions) ==========
print("\nCreating SHAP Force Plot for sample prediction...")

# Use first sample for force plot
base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    # In binary classification some shap versions return array of base values
    base_value = base_value[1] if len(base_value) > 1 else base_value[0]

plt.figure(figsize=(14, 4))
shap.force_plot(
    base_value,
    shap_values_pos[0],
    X_sample.iloc[0],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_shap_force_plot_sample.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved to: 04_shap_force_plot_sample.png")

# ========== SHAP Dependence Plot for ps_car_11_cat (Special) ==========
"""
print("\nCreating SHAP Dependence Plot for ps_car_11_cat (special focus)...")
if 'ps_car_11_cat' in X_sample.columns:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        'ps_car_11_cat',
        shap_values_pos,
        X_sample,
        feature_names=X_sample.columns,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_shap_dependence_special_ps_car_11_cat.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created special plot for ps_car_11_cat")


# ========== SHAP Waterfall Plot (Individual Predictions) ==========
print("Creating SHAP Waterfall Plot for sample prediction...")

explainer_beeswarm = shap.Explainer(model, X_sample)

# Use scalar or array expected value as needed
expected_value_beeswarm = explainer_beeswarm.expected_value
if isinstance(expected_value_beeswarm, (list, np.ndarray)):
    expected_value_beeswarm = expected_value_beeswarm[1] if len(expected_value_beeswarm) > 1 else expected_value_beeswarm[0]

plt.figure(figsize=(12, 6))
shap.plots._waterfall.waterfall_legacy(
    expected_value_beeswarm,
    shap_values_pos[0],
    X_sample.iloc[0],
    feature_names=X_sample.columns
)
plt.tight_layout()
plt.savefig(f'{output_dir}/05_shap_waterfall_plot_sample.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved to: 05_shap_waterfall_plot_sample.png")
"""

# ============================================================================
# SAVE SHAP VALUES FOR FUTURE USE
# ============================================================================

print("\nSaving SHAP values and sample data...")
np.save(f'{output_dir}/shap_values.npy', shap_values_pos)
X_sample.to_csv(f'{output_dir}/X_sample_for_shap.csv', index=False)
np.save(f'{output_dir}/y_sample_for_shap.npy', y_sample)

print("\n[OK] SHAP values saved for future analysis")

# ============================================================================
# CLEANUP MEMORY
# ============================================================================

print("\nCleaning up memory...")
del explainer, shap_values, shap_values_pos, X_sample, y_sample
gc.collect()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 03b COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"\nAll SHAP visualizations saved to: {output_dir}/")
print("\nGenerated files:")
print("  1. 01_shap_summary_dot.png       - Feature impact overview")
print("  2. 02_shap_summary_bar.png       - Feature importance ranking")
print("  3. shap_feature_importance.csv   - Numerical importance values")
print("  4. 03_shap_dependence_*.png      - Top 6 feature relationships")
print("  5. 04_shap_force_plot_sample.png - Individual prediction breakdown")
print("  6. 05_shap_waterfall_plot_sample.png - Decision path visualization")
print("  7. shap_values.npy               - Raw SHAP values for reuse")

