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
# for PDF outputs (must be set before importing pyplot)
mpl.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from metrics import bootstrap_gini_ci

# Embed fonts as TrueType (Type 42) so text stays real text / editable in PDF
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
ax.legend(loc='upper right')
plt.tight_layout()
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
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/high_correlation_pairs.pdf', bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/high_correlation_pairs.pdf")


# ============================================================================
# 6. CHI-SQUARE TEST + CRAMER'S V FOR EACH ps_calc_* COLUMN VS TARGET
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Chi-square test + Cramer's V (ps_calc_* vs. target)")
print("=" * 70)


def cramers_v(confusion_matrix):
    """Bias-corrected Cramer's V effect size for a contingency table."""
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((k_corr - 1), (r_corr - 1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2_corr / denom)


chi2_results = []
for col in ps_calc_cols:
    try:
        contingency = pd.crosstab(df[col], target)
        chi2_stat, p_value, _, _ = chi2_contingency(contingency)
        v = cramers_v(contingency)
        chi2_results.append({
            'feature': col,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'cramers_v': v,
            'significant_p<0.05': p_value < 0.05
        })
    except Exception as e:
        chi2_results.append({
            'feature': col, 'chi2_statistic': np.nan, 'p_value': np.nan,
            'cramers_v': np.nan, 'significant_p<0.05': False
        })
        print(f"  Warning: could not test {col}: {e}")

chi2_df = pd.DataFrame(chi2_results).sort_values('cramers_v', ascending=False)
chi2_df.to_csv(f'{OUTPUT_DIR}/ps_calc_chi2_cramers_v.csv', index=False)

print(chi2_df.to_string(index=False, formatters={
    'chi2_statistic': '{:.2f}'.format,
    'p_value': '{:.4f}'.format,
    'cramers_v': '{:.4f}'.format
}))
print(f"\nMean Cramer's V across ps_calc_*: {chi2_df['cramers_v'].mean():.4f}")
print(f"Max  Cramer's V across ps_calc_*: {chi2_df['cramers_v'].max():.4f}")
print(f"✓ Saved: {OUTPUT_DIR}/ps_calc_chi2_cramers_v.csv")


# ============================================================================
# 7. CORRELATION STRUCTURE AMONG ps_calc_* COLUMNS
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Correlation structure among ps_calc_* columns")
print("=" * 70)

calc_corr = df[ps_calc_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(calc_corr, cmap='coolwarm', center=0, square=True,
            cbar_kws={'label': 'Pearson correlation'})
plt.title("Correlation Matrix: ps_calc_* Features", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ps_calc_correlation_heatmap.pdf', bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/ps_calc_correlation_heatmap.pdf")


# ============================================================================
# 8. TARGET-CONDITIONED DISTRIBUTIONS (sample of ps_calc_* columns)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Target-conditioned distributions (sample columns)")
print("=" * 70)

sample_cols = ps_calc_cols[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), sample_cols):
    for t_val, color, label in [(0, 'steelblue', 'target=0'), (1, 'indianred', 'target=1')]:
        subset = df.loc[df['target'] == t_val, col]
        ax.hist(subset, bins=20, density=True, alpha=0.5, color=color, label=label)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)
plt.suptitle("Target-Conditioned Distributions: Sample ps_calc_* Features", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ps_calc_target_conditioned_distributions.pdf', bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/ps_calc_target_conditioned_distributions.pdf")


# ============================================================================
# 9. LIGHTGBM FEATURE IMPORTANCE — WITH ps_calc_* INCLUDED
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: LightGBM feature importance WITH ps_calc_* included")
print("=" * 70)

X_full = df.drop(columns=['id', 'target'])
y_full = df['target']

categorical_cols_full = [c for c in X_full.columns if c.endswith('_cat')]
for c in categorical_cols_full:
    X_full[c] = X_full[c].astype('category')

print("Training a single LightGBM model on ALL features (including ps_calc_*) "
      "to compare relative importances...")
model_full = lgb.LGBMClassifier(**LGBM_BASELINE_PARAMS)
model_full.fit(X_full, y_full, categorical_feature=categorical_cols_full)

importance_df = pd.DataFrame({
    'feature': X_full.columns,
    'importance_gain': model_full.booster_.feature_importance(importance_type='gain'),
    'is_ps_calc': [c.startswith('ps_calc') for c in X_full.columns]
}).sort_values('importance_gain', ascending=False).reset_index(drop=True)
importance_df['rank'] = importance_df.index + 1

importance_df.to_csv(f'{OUTPUT_DIR}/full_feature_importance_with_ps_calc.csv', index=False)

plt.figure(figsize=(10, 14))
colors = ['indianred' if c else 'steelblue' for c in importance_df['is_ps_calc']]
plt.barh(importance_df['feature'], importance_df['importance_gain'], color=colors)
plt.gca().invert_yaxis()
plt.xlabel('LightGBM Importance (Gain)', fontsize=11)
plt.title("Feature Importance — ps_calc_* (red) vs. Retained Features (blue)",
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/full_feature_importance_plot.pdf', bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/full_feature_importance_with_ps_calc.csv")
print(f"✓ Saved: {OUTPUT_DIR}/full_feature_importance_plot.pdf")

del model_full


# ============================================================================
# 10. ABLATION: 5-FOLD GINI WITH vs. WITHOUT ps_calc_* (BOOTSTRAP CI)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Ablation — Gini WITH vs. WITHOUT ps_calc_* (5-fold CV, Bootstrap CI)")
print("=" * 70)


def run_cv_gini(X, y, cat_cols, label, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    all_y_val, all_probs = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        cur_cat = [c for c in cat_cols if c in X_train.columns]
        for c in cur_cat:
            X_train[c] = X_train[c].astype('category')
            X_val[c] = X_val[c].astype('category')

        model = lgb.LGBMClassifier(**LGBM_BASELINE_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            categorical_feature=cur_cat if cur_cat else None,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
        )
        probs = model.predict_proba(X_val)[:, 1]
        all_y_val.extend(y_val.values)
        all_probs.extend(probs)
        fold_auc = roc_auc_score(y_val, probs)
        print(f"  [{label}] Fold {fold} | Gini: {2*fold_auc - 1:.5f}")
        del model

    boot_mean, lower, upper = bootstrap_gini_ci(all_y_val, all_probs)
    print(f"  [{label}] Bootstrap Gini: {boot_mean:.5f} (95% CI: [{lower:.5f}, {upper:.5f}])")
    return {'label': label, 'gini_mean': boot_mean, 'gini_ci_lower': lower, 'gini_ci_upper': upper}


X_with_calc = df.drop(columns=['id', 'target']).copy()
X_without_calc = X_with_calc.drop(columns=ps_calc_cols).copy()
cat_cols_all = [c for c in X_with_calc.columns if c.endswith('_cat')]

print("\nRunning ablation CV WITH ps_calc_*...")
result_with = run_cv_gini(X_with_calc, y_full, cat_cols_all, label='WITH ps_calc_*')

print("\nRunning ablation CV WITHOUT ps_calc_*...")
result_without = run_cv_gini(X_without_calc, y_full, cat_cols_all, label='WITHOUT ps_calc_*')

ablation_df = pd.DataFrame([result_with, result_without])
ablation_df.to_csv(f'{OUTPUT_DIR}/ps_calc_ablation_gini_comparison.csv', index=False)

print("\n" + "=" * 70)
print("ABLATION SUMMARY")
print("=" * 70)
print(ablation_df.to_string(index=False, formatters={
    'gini_mean': '{:.5f}'.format,
    'gini_ci_lower': '{:.5f}'.format,
    'gini_ci_upper': '{:.5f}'.format
}))
gini_diff = result_with['gini_mean'] - result_without['gini_mean']
print(f"\nGini(WITH) - Gini(WITHOUT) = {gini_diff:+.5f}")
ci_overlap = not (result_with['gini_ci_lower'] > result_without['gini_ci_upper'] or
                   result_without['gini_ci_lower'] > result_with['gini_ci_upper'])
print(f"95% CIs overlap: {ci_overlap}")
print(f"✓ Saved: {OUTPUT_DIR}/ps_calc_ablation_gini_comparison.csv")

print("\n" + "=" * 70)
print("PHASE 0 EDA COMPLETED — all evidence and vector plots saved to eda_results/")
print("=" * 70)