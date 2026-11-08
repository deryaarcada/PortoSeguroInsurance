"""
PHASE 0: EDA — Justifying the Removal of 'ps_calc_*' Features
================================================================
This is a standalone, exploratory script (NOT part of the core pipeline).
It runs on the RAW train.csv (before 01_Preprocessing.py drops the ps_calc
columns) and produces statistical + empirical evidence for why the
ps_calc_* features were excluded from modeling.

Run this ONCE, manually, before running the main pipeline. Its outputs are
meant to be embedded in the thesis (tables/figures), not consumed by any
downstream script.

Evidence produced:
  1. Target distribution analysis & visualization
  2. Missing values (-1 encoded) analysis & visualization
  3. Categorical feature cardinality and missing percentage check (Figure 3.4)
  4. High correlation pairs among ordinal/continuous features (Figure 3.5)
  5. Chi-square test + Cramer's V for each ps_calc_* column vs. target
  6. Correlation heatmap among ps_calc_* columns
  7. Target-conditioned distributions for sample ps_calc_* columns
  8. LightGBM feature importance WITH ps_calc_* included (colored plot)
  9. Ablation experiment: 5-fold CV Gini WITH vs. WITHOUT ps_calc_*

Outputs written to: eda_results/ (as vector PDF files for thesis embedding)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
# for PDF outputs
mpl.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from metrics import bootstrap_gini_ci

mpl.rcParams["pdf.fonttype"] = 42

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'eda_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

LGBM_BASELINE_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=2,
    verbosity=-1
)


# ============================================================================
# 1. LOAD RAW DATA (ps_calc_* still present)
# ============================================================================
print("=" * 70)
print("PHASE 0: EDA — Justifying Removal of ps_calc_* Features")
print("=" * 70)

print("\nLoading raw train.csv (ps_calc_* columns not yet dropped)...")
df = pd.read_csv("train.csv")
print(f"Train shape: {df.shape}")

target = df['target']
ps_calc_cols = [c for c in df.columns if c.startswith('ps_calc')]
other_cols = [c for c in df.columns if c not in ps_calc_cols + ['id', 'target']]

print(f"Found {len(ps_calc_cols)} ps_calc_* columns and {len(other_cols)} other feature columns.")


# ============================================================================
# 2. TARGET DISTRIBUTION ANALYSIS & PLOT
# ============================================================================
print("\n" + "=" * 70)
print("STEP 0.1: Target Distribution Analysis")
print("=" * 70)

target_counts = df['target'].value_counts()
target_props = df['target'].value_counts(normalize=True) * 100
target_summary = pd.DataFrame({'count': target_counts, 'percentage (%)': target_props})
print(target_summary)

fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(x=target_counts.index, y=target_counts.values, palette=['skyblue', 'orange'], ax=ax)
ax.set_title("Target Distribution (0=No Claim, 1=Claim Filed)", fontsize=11, fontweight='bold')
ax.set_xlabel("target", fontsize=10)
ax.set_ylabel("count", fontsize=10)
plt.tight_layout()

plt.savefig(f'{OUTPUT_DIR}/target_distribution.pdf', bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/target_distribution.pdf")


# ============================================================================
# 3. MISSING VALUES ANALYSIS (-1 encoded values)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 0.2: Missing Values Analysis (-1 encoded)")
print("=" * 70)

missing_counts = (df == -1).sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
missing_pcts = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({'Missing count': missing_counts, 'Missing %': missing_pcts})
print(missing_df.to_string())

fig, ax = plt.subplots(figsize=(9, 6))
missing_pcts.plot(kind='barh', color='skyblue', ax=ax)
ax.set_title("Missing (-1) Value Percentage by Feature", fontsize=11, fontweight='bold')
ax.set_xlabel("% Missing", fontsize=10)
ax.invert_yaxis()
# for latex 
#ax.set_yticklabels([label.get_text().replace('_', r'\_') for label in ax.get_yticklabels()])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/missing_values_analysis.pdf', bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/missing_values_analysis.pdf")



# ============================================================================
# 4. CATEGORICAL FEATURE STATISTICS: CARDINALITY AND MISSING % (Figure 3.4)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 0.3: Categorical Feature Statistics (Cardinality & Missing %)")
print("=" * 70)

cat_cols = [c for c in df.columns if c.endswith('_cat')]
cat_cardinality = df[cat_cols].nunique()
cat_missing_pct = (df[cat_cols] == -1).mean() * 100
cat_stats_df = pd.DataFrame({
    'Cardinality': cat_cardinality,
    'Missing %': cat_missing_pct
}).sort_values('Cardinality', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cat_cols))
width = 0.4

# Sort columns by cardinality descending for a cleaner plot matching Figure 3.4
sorted_cat_cols = cat_stats_df.index
sorted_cardinality = cat_stats_df['Cardinality']
sorted_missing = cat_stats_df['Missing %']

ax.bar(np.arange(len(sorted_cat_cols)) - width/2, sorted_cardinality, width, label='Cardinality', color='skyblue')
ax.bar(np.arange(len(sorted_cat_cols)) + width/2, sorted_missing, width, label='Missing %', color='orange')

ax.set_xticks(np.arange(len(sorted_cat_cols)))
ax.set_xticklabels(sorted_cat_cols, rotation=90)
ax.set_title("Categorical Feature Statistics: Cardinality and Missing %", fontsize=11, fontweight='bold')
ax.set_ylabel("Count / Cardinality", fontsize=10)
#ax.set_xticklabels([label.get_text().replace('_', r'\_') for label in ax.get_xticklabels()])
#ax.set_yticklabels([label.get_text().replace('_', r'\_') for label in ax.get_yticklabels()])
ax.legend(loc='upper right')
plt.tight_layout()
# .svg yerine .pdf olarak kaydediliyor
plt.savefig(f'{OUTPUT_DIR}/categorical_cardinality_missing.pdf', bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/categorical_cardinality_missing.pdf")


# ============================================================================
# 5. HIGH CORRELATION PAIRS (ORDINAL / CONTINUOUS) (Figure 3.5)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 0.4: High Correlation Pairs (Ordinal / Continuous)")
print("=" * 70)

# Select ordinal and continuous features as displayed in Figure 3.5
ordinal_continuous_cols = [c for c in df.columns if ('ind' in c or 'reg' in c or 'car' in c or 'calc' in c) and not c.endswith('_cat') and not c.endswith('_bin') and c != 'id']
# Limit to a representative subset if too large, or use specific columns matching Fig 3.5 sample
fig_3_5_cols = [c for c in df.columns if c in ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01', 'ps_reg_02', 'ps_car_11', 'ps_car_15'] + ps_calc_cols[:14]]
existing_fig_cols = [c for c in fig_3_5_cols if c in df.columns]

corr_subset = df[existing_fig_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})
ax.set_title("High correlation pairs (ordinal):", fontsize=11, fontweight='bold', loc='left')
#ax.set_xticklabels([label.get_text().replace('_', r'\_') for label in ax.get_xticklabels()])
#ax.set_yticklabels([label.get_text().replace('_', r'\_') for label in ax.get_yticklabels()])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/high_correlation_pairs.pdf', bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/high_correlation_pairs.pdf")
