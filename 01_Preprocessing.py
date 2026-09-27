"""
PHASE 1: EDA, Preprocessing & Feature Engineering 
=====================================================================
This script handles:
- Data loading and initial EDA
- Data preprocessing and cleaning
- Feature engineering
- Saves preprocessed data to disk for Phase 2

"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import joblib
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

print("=" * 70)
print("PHASE 1: EDA, Preprocessing & Feature Engineering")
print("=" * 70)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)

output_dir = 'preprocessing_cache'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
# %%
print("\n--- Loading Data ---")
df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Train shape: {df.shape}")
print(f"Test shape: {test_df.shape}")

# Basic info
print(f"\nTarget distribution (Train):")
print(df['target'].value_counts(normalize=True))

# ============================================================================
# 2. FEATURE CATEGORIZATION
# ============================================================================
# %%
binary_cols = [col for col in df.columns if col.endswith('_bin')]
categorical_cols = [col for col in df.columns if col.endswith('_cat')]
ps_calc_cols = [col for col in df.columns if col.startswith('ps_calc')]
continuous_cols = [col for col in df.columns if col not in binary_cols + categorical_cols + ['id', 'target']]

print(f"\nFeature Types:")
print(f"  Binary features:     {len(binary_cols)}")
print(f"  Categorical features: {len(categorical_cols)}")
print(f"  Continuous features: {len(continuous_cols)}")
print(f"  Calculation features (to drop): {len(ps_calc_cols)}")

# ============================================================================
# 3. DATA CLEANING
# ============================================================================

print("\n--- Data Cleaning ---")

# Drop ID column from train
X = df.drop(columns=['id', 'target'])
y = df['target']

# Drop noisy ps_calc columns
X = X.drop(columns=ps_calc_cols)
print(f"Dropped {len(ps_calc_cols)} ps_calc columns")

# Handle missing values (for train)
print(f"\nMissing values in train:\n{X.isnull().sum()[X.isnull().sum() > 0]}")

# Drop observations with missing values (minimal impact in this dataset)
if X.isnull().sum().sum() > 0:
    X = X.dropna()
    y = y.loc[X.index]
    print(f"Dropped {len(df) - len(X)} rows with missing values")

print(f"Final train shape: {X.shape}, {y.shape}")

# ============================================================================
# 4. CATEGORICAL ENCODING (Ensure proper types for LightGBM)
# ============================================================================
# %%
print("\n--- Categorical Encoding ---")

# Update categorical cols after dropping ps_calc
categorical_cols = [col for col in categorical_cols if col in X.columns]
continuous_cols = [col for col in continuous_cols if col in X.columns]  # Update continuous cols too
current_cat_features = [c for c in categorical_cols if c != 'ps_car_11_cat']

# Convert to category dtype for better memory efficiency
for col in current_cat_features:
    X[col] = X[col].astype('category')

print(f"Converted {len(current_cat_features)} columns to 'category' dtype")


# ============================================================================
# Rare-category candidate columns (NOTE: the actual mapping is NOT learned
# here anymore — see explanation below)
# ============================================================================
# Identify ordinal-like columns (heuristic: column name contains 'ord')
ordinal_cols = [c for c in X.columns if 'ord' in c]

# Candidate columns to check for rare classes: categorical and ordinal columns
cols_for_rare = [c for c in categorical_cols if c in X.columns] + [c for c in ordinal_cols if c in X.columns and c not in categorical_cols]

# ----------------------------------------------------------------------
# FOLD-SAFE PREPROCESSING FIX:
# This script used to learn the rare-category mapping (which values are
# rare, i.e. <1% frequency) from the FULL training set right here, before
# any cross-validation split existed downstream. That is not target
# leakage (no label information is involved), but it does mean each CV
# validation fold's rows helped decide which categories count as "rare"
# for the very fold they are later held out from, a mild preprocessing
# leakage.
#
# The fix: only the CANDIDATE column list (`cols_for_rare`, just column
# names — no data statistics) is decided here and saved for downstream
# scripts to use. The actual thresholding (which specific values are rare)
# is now learned separately, INSIDE each CV fold's training split, in
# 02a_BaselineModels.py, 02b_HyperparameterTuning.py, and
# 03a_ModelTraining_Bootstrap.py, via the shared fit_rare_map() /
# apply_rare_map() helpers in preprocessing_utils.py. X_preprocessed.csv
# and test_preprocessed.csv below therefore do NOT have rare categories
# collapsed yet, that happens fold-by-fold downstream.
# ----------------------------------------------------------------------
print(f"\nIdentified {len(cols_for_rare)} candidate columns for rare-category "
      f"mapping (categorical + ordinal). Thresholding is deferred to each "
      f"downstream CV fold — see preprocessing_utils.py.")

# ============================================================================
# 5. FEATURE SCALING 
# ============================================================================

print("\n--- Feature Statistics ---")
print(f"\nContinuous features - min/max/mean:")
print(X[continuous_cols].describe().loc[['min', 'max', 'mean']].round(3))

# ============================================================================
# 6. TEST SET PREPROCESSING
# ============================================================================

print("\n--- Processing Test Set ---")

test_features = test_df.drop(columns=['id'])
test_ids = test_df['id']

# Drop ps_calc columns
test_features = test_features.drop(columns=ps_calc_cols)

# Ensure column order matches training set
test_features = test_features[X.columns]

# NOTE: rare-category mapping is intentionally NOT applied here anymore.
# Each fold's model in the downstream CV loops (02a/02b/03a) fits its own
# rare-category map on that fold's training partition and applies it to
# both that fold's validation rows and (in 03a) the test set, so the
# mapping used to score `test_features` is always consistent with the
# mapping the corresponding model was trained under. See preprocessing_utils.py.

# Convert categorical columns to category dtype
for col in current_cat_features:
    test_features[col] = test_features[col].astype('category')

print(f"Test features shape: {test_features.shape}")
print(f"Columns match training set: {list(X.columns) == list(test_features.columns)}")

# ============================================================================
# 7. SAVE PREPROCESSED DATA
# ============================================================================

print("\n--- Saving Preprocessed Data ---")

X.to_csv(f'{output_dir}/X_preprocessed.csv', index=False)
pd.DataFrame(y).to_csv(f'{output_dir}/y_preprocessed.csv', index=False)
test_features.to_csv(f'{output_dir}/test_preprocessed.csv', index=False)

# Save metadata
metadata = {
    'binary_cols': binary_cols,
    'categorical_cols': categorical_cols,
    'continuous_cols': continuous_cols,
    'current_cat_features': current_cat_features,
    'cols_for_rare': cols_for_rare,
    'X_shape': X.shape,
    'y_shape': y.shape,
    'test_shape': test_features.shape
}
joblib.dump(metadata, f'{output_dir}/metadata.pkl')
joblib.dump(categorical_cols, f'{output_dir}/categorical_cols.pkl')
joblib.dump(cols_for_rare, f'{output_dir}/cols_for_rare.pkl')

print(f"✓ X_preprocessed.csv ({X.shape})")
print(f"✓ y_preprocessed.csv ({y.shape})")
print(f"✓ test_preprocessed.csv ({test_features.shape})")
print(f"✓ metadata.pkl")
print(f"✓ categorical_cols.pkl")
print(f"✓ cols_for_rare.pkl (candidate columns only — thresholding done fold-by-fold downstream)")

# ============================================================================
# 8. CREATE SYMLINKS FOR PHASE 2 
# ============================================================================

print("\n--- Creating convenience copies ---")

import shutil

# Copy to root for easy access in Phase 2
shutil.copy(f'{output_dir}/X_preprocessed.csv', 'X_preprocessed.csv')
shutil.copy(f'{output_dir}/y_preprocessed.csv', 'y_preprocessed.csv')
shutil.copy(f'{output_dir}/test_preprocessed.csv', 'test_preprocessed.csv')
shutil.copy(f'{output_dir}/categorical_cols.pkl', 'categorical_cols.pkl')
shutil.copy(f'{output_dir}/cols_for_rare.pkl', 'cols_for_rare.pkl')

print("✓ Copied files to root directory for Phase 2")

# ============================================================================
# 9. MEMORY CLEANUP
# ============================================================================

print("\n--- Memory Cleanup ---")
del df, test_df, X, y, test_features
gc.collect()

print("✓ Memory cleared")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 1 COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nPreprocessed data saved. You can now run Phase 2:")
print("  python 02a_BaselineModels.py")


# %%
