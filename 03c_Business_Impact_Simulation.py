"""
PHASE 3B: Model Interpretability with SHAP (NumPy Array Safe Mode)
======================================================================
This script handles SHAP calculations safely by converting data to pure 
NumPy arrays to strictly bypass LightGBM's rigid pandas categorical validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import shap
import os

warnings.filterwarnings('ignore')

print("=" * 70)
print("PHASE 3B: SHAP Interpretability Analysis (NumPy Mode)")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA AND MODELS
# ============================================================================
print("\nLoading preprocessed data and trained models...")

if not os.path.exists('X_preprocessed.csv') or not os.path.exists('y_preprocessed.csv'):
    raise FileNotFoundError("Preprocessed data files missing. Please run preprocessing first.")

X = pd.read_csv('X_preprocessed.csv')
y = pd.read_csv('y_preprocessed.csv').iloc[:, 0]

final_model_path = 'model_results/final_model.pkl'
if not os.path.exists(final_model_path):
    raise FileNotFoundError("final_model.pkl not found under 'model_results/'.")

final_model = joblib.load(final_model_path)
print('✓ Loaded final production model.')

# ============================================================================
# 2. EXTRACT BOOSTER & ALIGN FEATURES
# ============================================================================
if hasattr(final_model, "booster_"):
    booster = final_model.booster_
else:
    booster = final_model

# Ensure columns match model's expected internal sequence
if hasattr(final_model, 'feature_name_'):
    feature_names = final_model.feature_name_
    X = X[feature_names]
elif hasattr(booster, 'feature_name'):
    feature_names = booster.feature_name()
    X = X[feature_names]
else:
    feature_names = X.columns.tolist()

# ============================================================================
# 3. SAMPLING & PURE NUMPY CONVERSION (The Ultimate Bypass)
# ============================================================================
print("\nPreparing sample data for SHAP analysis (1000 random samples)...")

sample_size = min(1000, len(X))
np.random.seed(42)
sample_indices = np.random.choice(len(X), size=sample_size, replace=False)

X_sample_df = X.iloc[sample_indices].copy()

# Step A: Enforce raw numeric encodings on any remaining object/category artifacts
for col in X_sample_df.columns:
    if X_sample_df[col].dtype == 'object' or isinstance(X_sample_df[col].dtype, pd.CategoricalDtype):
        X_sample_df[col] = X_sample_df[col].astype('category').cat.codes
    X_sample_df[col] = pd.to_numeric(X_sample_df[col], errors='coerce').fillna(-1)

# Step B: CRITICAL STEP - Strip Pandas identity completely by extracting raw NumPy representation
X_sample_numpy = X_sample_df.values

print(f"✓ Data converted to pure NumPy array. Matrix shape: {X_sample_numpy.shape}")

# ============================================================================
# 4. CALCULATE SHAP VALUES
# ============================================================================
print("\nCalculating SHAP values...")
print("(Executing inside C++ tree structure via NumPy matrix...)\n")

# TreeExplainer using pure array skips categorical_feature verification checks
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X_sample_numpy)

# Parse output array matrices based on SHAP return format
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1] if len(shap_values) > 1 else shap_values[0]
else:
    if len(shap_values.shape) == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

print("[OK] SHAP values successfully calculated.\n")

# ============================================================================
# 5. SHAP VISUALIZATIONS (Mapping back names for plotting only)
# ============================================================================
output_dir = 'model_results/shap_analysis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a clean plotting DataFrame so that the charts show feature names instead of integers
plot_df = pd.DataFrame(X_sample_numpy, columns=feature_names)

# ========== SHAP Summary Plot (Dot) ==========
print("Creating SHAP Summary Plot (Dot)...")
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_pos,
    plot_df,
    plot_type="dot",
    show=False,
    max_display=20
)
plt.title("SHAP Summary Plot: Feature Impact on Model Output", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_shap_summary_dot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved to: {output_dir}/01_shap_summary_dot.png")

# ========== SHAP Summary Plot (Bar) ==========
print("Creating SHAP Summary Plot (Bar)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values_pos,
    plot_df,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.title("SHAP Mean Absolute Impact on Model Output", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved to: {output_dir}/02_shap_summary_bar.png")

print("\n=" * 70)
print("SHAP ANALYSIS PHASE COMPLETED SUCCESSFULLY!")
print("=" * 70)
