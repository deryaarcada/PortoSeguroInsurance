"""
PHASE 3b: Model Interpretability with SHAP
==========================================

Run AFTER Phase 3a has generated model results.
Although Porto Seguro data is anonymized, the purpose of the features can be partially understood. In this respect, SHAP analysis is performed in section 3b.

This script:
- Loads the final production LightGBM model from Phase 3a
- Applies the same feature representation used in Phase 3a
- Computes SHAP values on a reproducible sample of 1,000 observations
- Generates SHAP summary, importance, dependence, force and waterfall plots
- Saves SHAP values and supporting data

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gc
import warnings
import shap
import os

warnings.filterwarnings('ignore')


# ============================================================================
# LOAD DATA AND FINAL MODEL FROM 03A
# ============================================================================

print("=" * 70)
print("PHASE 3b: SHAP Interpretability Analysis")
print("=" * 70)

print("\nLoading preprocessed data...")

X = pd.read_csv('X_preprocessed.csv')
y = pd.read_csv('y_preprocessed.csv').iloc[:, 0]


# ============================================================================
# LOAD FINAL MODEL CREATED IN 03A
# ============================================================================

final_model_path = 'model_results/final_model.pkl'

if not os.path.exists(final_model_path):
    raise FileNotFoundError(
        "final_model.pkl not found. "
        "Run 03a_ModelTraining_Bootstrap.py first."
    )

print("\nLoading FINAL production model from Phase 03a...")

final_model = joblib.load(final_model_path)

print("\nMODEL LOADED FROM:")
print(final_model_path)

print("Model type:", type(final_model))
print(
    "n_estimators_:",
    getattr(final_model, "n_estimators_", None)
)
print(
    "Best iteration:",
    getattr(final_model, "best_iteration_", None)
)

try:
    print(
        "Num trees:",
        final_model.booster_.num_trees()
    )
except Exception:
    print("Could not access booster tree count.")


# ============================================================================
# APPLY SAME FEATURE REPRESENTATION USED IN 03A
# ============================================================================

print("\nApplying the same feature representation used in Phase 03a...")



try:
    categorical_cols = joblib.load('categorical_cols.pkl')
except Exception:
    categorical_cols = []

current_cat_features = [
    c for c in categorical_cols
    if c in X.columns
]

for col in current_cat_features:
    X[col] = X[col].astype('category')

print(
    f"Converted {len(current_cat_features)} "
    "categorical columns to 'category' dtype."
)

if 'ps_car_11_cat' in X.columns:
    print(
        "ps_car_11_cat handling: "
        "native LightGBM categorical feature "
        "(NO target encoding)"
    )


print(f"\nX shape: {X.shape}")
print("Final model successfully loaded.")


# ============================================================================
# VALIDATE FEATURE COMPATIBILITY
# ============================================================================

print("\nValidating feature compatibility...")

model_features = list(final_model.feature_name_)

print("MODEL FEATURE COUNT:", len(model_features))
print("DATA FEATURE COUNT :", len(X.columns))

assert len(model_features) == X.shape[1], (
    f"Feature count mismatch: "
    f"model={len(model_features)}, data={X.shape[1]}"
)

# The model's feature order is the source of truth.
X = X[model_features]

assert list(X.columns) == model_features

print("✓ Feature count matches.")
print("✓ Feature order matches final model.")


# ============================================================================
# PREPARE SAMPLE DATA FOR SHAP
# ============================================================================

print(
    "\nPreparing sample data for SHAP analysis "
    "(1000 random samples)..."
)

sample_size = min(1000, len(X))

np.random.seed(42)

sample_indices = np.random.choice(
    len(X),
    size=sample_size,
    replace=False
)

X_sample = X.iloc[sample_indices].copy()

y_sample = y.iloc[sample_indices].values

feature_names = list(X_sample.columns)

print(
    f"✓ Selected {sample_size} observations "
    "using random_state=42."
)


# ============================================================================
# CONVERT SAMPLE TO NUMPY
# ============================================================================

print("\nConverting SHAP input to NumPy...")

X_sample_np = X_sample.to_numpy(dtype=np.float64)

print(
    f"✓ Sample ready: "
    f"{X_sample_np.shape[0]} rows × "
    f"{X_sample_np.shape[1]} features"
)


print("\nFIRST 10 MODEL FEATURES:")
print(model_features[:10])

print("\nFIRST 10 SHAP FEATURES:")
print(feature_names[:10])

assert feature_names == model_features

print("✓ Model and SHAP feature order verified.")


# ============================================================================
# CALCULATE SHAP VALUES
# ============================================================================

print("\nCalculating SHAP values...")
print(
    "(This may take 1-3 minutes depending on your hardware)\n"
)

explainer = shap.TreeExplainer(
    final_model.booster_
)

# Pass NumPy array to avoid pandas dtype handling issues.
shap_values = explainer.shap_values(
    X_sample_np
)


# ============================================================================
# EXTRACT POSITIVE-CLASS SHAP VALUES
# ============================================================================

# Binary classification:
# depending on SHAP version, shap_values may be:
#   - list [class_0, class_1]
#   - single NumPy array

if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
else:
    shap_values_pos = shap_values

print("✓ SHAP values calculated.\n")


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

output_dir = 'model_results/shap_analysis'

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================================
# SHAP VISUALIZATIONS
# ============================================================================

# ============================================================================
# 1. SUMMARY PLOT - DOT
# ============================================================================

print("Creating SHAP Summary Plot (Dot)...")

plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values_pos,
    X_sample_np,
    feature_names=feature_names,
    plot_type="dot",
    show=False,
    max_display=len(feature_names)
)

plt.title(
    "SHAP Summary Plot: Feature Impact on Model Output",
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    f'{output_dir}/01_shap_summary_dot.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("[OK] Saved: 01_shap_summary_dot.png")


# ============================================================================
# 2. SUMMARY PLOT - BAR
# ============================================================================

print("Creating SHAP Summary Plot (Bar)...")

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values_pos,
    X_sample_np,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=len(feature_names)
)

plt.title(
    "SHAP Mean Absolute Impact on Model Output",
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    f'{output_dir}/02_shap_summary_bar.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("[OK] Saved: 02_shap_summary_bar.png")


# ============================================================================
# 3. FEATURE IMPORTANCE TABLE
# ============================================================================

print("Creating Feature Importance Table...")

mean_abs_shap = np.abs(
    shap_values_pos
).mean(axis=0)

shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean |SHAP|': mean_abs_shap,
    'Mean SHAP': shap_values_pos.mean(axis=0)
}).sort_values(
    'Mean |SHAP|',
    ascending=False
)

print("\nTop 15 Most Important Features (by SHAP):")

print(
    shap_importance_df
    .head(15)
    .to_string(index=False)
)

shap_importance_df.to_csv(
    f'{output_dir}/shap_feature_importance.csv',
    index=False
)

print(
    "[OK] Saved: "
    "shap_feature_importance.csv"
)


# ============================================================================
# 4. TOP 20 FEATURE IMPORTANCE BAR PLOT
# ============================================================================

print(
    "\nCreating Top 20 Feature Importance Bar Plot..."
)

top_20_df = shap_importance_df.head(20)

plt.figure(figsize=(10, 8))

plt.barh(
    range(len(top_20_df)),
    top_20_df['Mean |SHAP|'].values[::-1]
)

plt.yticks(
    range(len(top_20_df)),
    top_20_df['Feature'].values[::-1]
)

plt.xlabel(
    'Mean |SHAP value|',
    fontsize=12
)

plt.title(
    'Top 20 Features by SHAP Importance',
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    f'{output_dir}/02b_shap_top20_bar.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(
    "[OK] Saved: "
    "02b_shap_top20_bar.png"
)


# ============================================================================
# 5. DEPENDENCE PLOTS - TOP 6 FEATURES
# ============================================================================

print(
    "\nCreating SHAP Dependence Plots "
    "for top 6 features..."
)

top_6_features = (
    shap_importance_df
    .head(6)['Feature']
    .values
)

for idx, feature in enumerate(
    top_6_features,
    1
):

    feat_idx = feature_names.index(feature)

    plt.figure(figsize=(10, 6))

    shap.dependence_plot(
        feat_idx,
        shap_values_pos,
        X_sample_np,
        feature_names=feature_names,
        show=False
    )

    safe_name = (
        feature
        .replace('/', '_')
        .replace('\\', '_')
    )

    plt.tight_layout()

    plt.savefig(
        f'{output_dir}/'
        f'03_shap_dependence_'
        f'{idx:02d}_{safe_name}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(
        f"  [OK] {idx}/6: {feature}"
    )


# ============================================================================
# 6. FORCE PLOT - SINGLE PREDICTION
# ============================================================================

print(
    "\nCreating SHAP Force Plot "
    "for sample prediction..."
)

base_value = explainer.expected_value

if isinstance(
    base_value,
    (list, np.ndarray)
):
    base_value = float(
        base_value[
            1 if len(base_value) > 1 else 0
        ]
    )
else:
    base_value = float(base_value)

plt.figure(figsize=(14, 4))

shap.force_plot(
    base_value,
    shap_values_pos[0],
    X_sample_np[0],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)

plt.tight_layout()

plt.savefig(
    f'{output_dir}/'
    '04_shap_force_plot_sample.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(
    "[OK] Saved: "
    "04_shap_force_plot_sample.png"
)


# ============================================================================
# 7. WATERFALL PLOT - SINGLE PREDICTION
# ============================================================================

print(
    "\nCreating SHAP Waterfall Plot "
    "for sample prediction..."
)

try:

    explanation = shap.Explanation(
        values=shap_values_pos[0],
        base_values=base_value,
        data=X_sample_np[0],
        feature_names=feature_names
    )

    plt.figure(figsize=(12, 6))

    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f'{output_dir}/'
        '05_shap_waterfall_plot_sample.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(
        "[OK] Saved: "
        "05_shap_waterfall_plot_sample.png"
    )

except Exception as e:

    print(
        f"⚠ Modern waterfall failed ({e}), "
        "trying legacy..."
    )

    try:

        plt.figure(figsize=(12, 6))

        shap.plots._waterfall.waterfall_legacy(
            base_value,
            shap_values_pos[0],
            X_sample_np[0],
            feature_names=feature_names
        )

        plt.tight_layout()

        plt.savefig(
            f'{output_dir}/'
            '05_shap_waterfall_plot_sample.png',
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(
            "[OK] Saved: "
            "05_shap_waterfall_plot_sample.png "
            "(legacy)"
        )

    except Exception as e2:

        print(
            f"⚠ Waterfall plot skipped: {e2}"
        )


# ============================================================================
# SAVE SHAP VALUES
# ============================================================================

print(
    "\nSaving SHAP values and sample data..."
)

np.save(
    f'{output_dir}/shap_values.npy',
    shap_values_pos
)

np.save(
    f'{output_dir}/X_sample_for_shap.npy',
    X_sample_np
)

np.save(
    f'{output_dir}/y_sample_for_shap.npy',
    y_sample
)

# Also save as CSV with column names.

pd.DataFrame(
    X_sample_np,
    columns=feature_names
).to_csv(
    f'{output_dir}/X_sample_for_shap.csv',
    index=False
)

print(
    "✓ SHAP values saved for future analysis"
)


# ============================================================================
# CLEANUP
# ============================================================================

print("\nCleaning up memory...")

del (
    explainer,
    shap_values,
    shap_values_pos,
    X_sample_np,
    y_sample
)

gc.collect()


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 03b COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nAll outputs saved to: {output_dir}/"
)

print("\nGenerated files:")

print(
    "  1. 01_shap_summary_dot.png"
    "           - Feature impact overview"
)

print(
    "  2. 02_shap_summary_bar.png"
    "           - Feature importance ranking"
)

print(
    "  3. 02b_shap_top20_bar.png"
    "            - Top 20 bar chart"
)

print(
    "  4. shap_feature_importance.csv"
    "       - Importance values table"
)

print(
    "  5. 03_shap_dependence_*.png"
    "          - Top 6 feature relationships"
)

print(
    "  6. 04_shap_force_plot_sample.png"
    "     - Single prediction breakdown"
)

print(
    "  7. 05_shap_waterfall_plot_sample.png"
    " - Decision path visualization"
)

print(
    "  8. shap_values.npy"
    "                   - Raw SHAP values"
)

print(
    "  9. X_sample_for_shap.csv/.npy"
    "        - Sample used for SHAP"
)