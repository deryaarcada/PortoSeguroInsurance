# End-to-End Car Insurance Claim Prediction Pipeline with Probability Calibration

An advanced, production-grade machine learning pipeline designed to predict automobile insurance claim probabilities under extreme class imbalance (~3.6% positive rate). Built on top of the Porto Seguro's Safe Driver Prediction benchmark, this project implements rigorous data preprocessing, leakage-free cross-validation, hyperparameter optimization, probability calibration, bootstrap uncertainty estimation, SHAP-driven interpretability, and operational business impact simulations.

## Key Achievements & Validation Benchmarks

* **Independent External Validation (Kaggle):** Achieved a **Public Gini Score of 0.27632** and a **Private Gini Score of 0.28394**. The higher private leaderboard score suggests good generalization performance and limited evidence of overfitting.
* **Leakage-Free Reliability:** The internal 5-Fold Stratified Cross-Validation Gini (0.27679, Bootstrap Mean, 95% CI: [0.26918, 0.28388]) sits close to the external Kaggle public benchmark, validating the integrity of the evaluation pipeline.
* **Actuarial-Grade Calibration:** Post-calibration Expected Calibration Error (ECE) was minimized using Isotonic Regression, ensuring predicted probabilities mirror empirical risk frequencies. Bootstrap-estimated ECE: 0.00030 (95% CI: [0.00008, 0.00067]); fold-averaged ECE: 0.00044. Brier Score: 0.03476.
* *__Evidence-Based Feature Reduction:__ The exclusion of the 20 `ps_calc_*` features is supported by a dedicated EDA study (see Phase 0 below), not just heuristic judgment — chi-square tests showed no significant association with the target (all p > 0.05), Cramér's V averaged 0.0004, and a controlled ablation experiment found no statistically detectable Gini difference with vs. without these features (ΔGini = +0.0005, fully overlapping 95% CIs).*

| Metric | Value |
| :--- | :--- |
| Kaggle Public Gini | 0.27632 |
| Kaggle Private Gini | 0.28394 |
| Internal CV Gini (Bootstrap Mean) | 0.27679 (95% CI: 0.26918–0.28388) |
| PR-AUC | 0.06645 |
| Brier Score | 0.03476 |
| Expected Calibration Error (ECE) | 0.00030 (bootstrap) / 0.00044 (fold-averaged) |

*Note: All internal metrics above are computed on out-of-fold (OOF) predictions from the final, leakage-free 5-fold pipeline described below.*

---

## Project Architecture & Execution Flow

The pipeline is modularized into dedicated Python scripts, enforcing clean separation of concerns and reproducibility. The pipeline is orchestrated end-to-end via `run_pipeline.py`, which runs 6 stages in sequence and logs timing/status for each. *An additional, standalone EDA script (Phase 0) is run separately, outside `run_pipeline.py`, to produce evidence for feature-removal decisions.*

### *0. Exploratory Data Analysis — Feature Removal Justification (optional, standalone)*
* ***`00_EDA_ps_calc_justification.py`***
  * *Runs once, manually, on the raw `train.csv` — before `01_Preprocessing.py` drops any columns — and is intentionally excluded from `run_pipeline.py` since it is exploratory/evidentiary rather than part of the reproducible modeling path.*
  * *Two categories of columns are removed prior to feature engineering, for distinct reasons:*
    * ***Structural removal:*** *the `id` column is dropped as a non-informative row identifier, and `target` is separated out as the prediction label to prevent trivial leakage if retained among the features.*
    * ***Evidence-based removal:*** *the 20 `ps_calc_*` columns are dropped based on empirical analysis rather than assumption. This script produces that evidence:*
      1. *Chi-square independence test + Cramér's V effect size for each `ps_calc_*` column vs. `target` (mean Cramér's V = 0.0004, max = 0.0029; no column significant at p < 0.05).*
      2. *Inter-correlation heatmap among `ps_calc_*` columns (mean \|r\| = 0.0009), indicating noise-like, uncorrelated structure.*
      3. *Target-conditioned distribution plots for a sample of `ps_calc_*` columns.*
      4. *A full-feature LightGBM importance ranking showing 17 of 20 `ps_calc_*` columns fall in the bottom half (mean rank 37.5 vs. 24.4 for retained features, out of 57 total).*
      5. *A controlled ablation experiment: 5-fold CV Gini with vs. without `ps_calc_*`, with 200-iteration bootstrap 95% CIs. Result: Gini(with) = 0.26948 [0.26161, 0.27737] vs. Gini(without) = 0.26903 [0.26127, 0.27707] — fully overlapping CIs, i.e. no statistically detectable difference.*
  * **Outputs Written To:** `eda_results/ps_calc_chi2_cramers_v.csv`, `eda_results/ps_calc_correlation_heatmap.png`, `eda_results/ps_calc_target_conditioned_distributions.png`, `eda_results/full_feature_importance_with_ps_calc.csv`, `eda_results/full_feature_importance_plot.png`, `eda_results/ps_calc_ablation_gini_comparison.csv`

### 1. Data Engineering & Pipeline Safety
* **`01_Preprocessing.py`**
  * Handles missing value imputation strategies customized for high-cardinality categorical data and continuous risk indices.
  * Drops the 20 noisy `ps_calc` columns *(justified empirically in Phase 0 above)* and casts categorical variables to Pandas `category` dtype.
  * Implements strict, training-derived rare-class mapping to structurally prevent data leakage between train and test.
  * Outputs `X_preprocessed.csv`, `y_preprocessed.csv`, `test_preprocessed.csv`, `metadata.pkl`, `categorical_cols.pkl`, `rare_class_map.pkl`, and copies convenience duplicates to the root directory for later phases.

### 2. Baseline Construction & Optimization
* **`02a_BaselineModels.py`**
  * Benchmarks LightGBM, Random Forest, and XGBoost out-of-the-box, using ALL preprocessed features (no upfront feature-selection step), via 5-Fold Stratified Cross-Validation.
  * This "all-features" design deliberately avoids a feature-selection-leakage pitfall that would occur if a Top-N feature list were derived from the full dataset and then reused across CV folds.
  * Reports Gini, PR-AUC, and Recall@1% per fold plus a Bootstrap 95% CI per model.
  * Current baseline result: LightGBM Gini 0.27975 [0.2713–0.2869], Random Forest 0.24895 [0.2413–0.2555], XGBoost 0.27510 [0.2679–0.2829].
  * Saves `model_results/baseline_metrics_backup.pkl` for later use in the ablation study.
  * (A separate, optional descriptive-only feature-importance script exists for exploratory reporting but is not part of the active `run_pipeline.py` sequence, since its output is not used to filter or select features downstream.)
* **`02b_HyperParameterTuning.py`**
  * Executes a 30-trial Bayesian hyperparameter search via Optuna (TPE sampler), optimizing `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, and `learning_rate`, using 5-Fold Stratified CV with early stopping.
  * Current best result: CV Gini 0.28300, with an average early-stopped tree count of 347 (search upper bound: n_estimators=500).
  * Also records `n_estimators_effective` — the average early-stopped tree count for the best trial — so Phase 3a can train the final production model with a tree count consistent with the reported tuning Gini, instead of the (larger) search-space upper bound.
  * **Outputs Written To:** `model_results/best_lgbm_params.pkl`

### 3. Robust Training, Validation, Interpretability & Business Alignment
* **`03a_ModelTRaining_Bootstrap.py`**
  * Trains the tuned LightGBM model across 5 Stratified folds using the Optuna-optimized hyperparameters.
  * Integrates a **fold-safe 80/20 calibration split** (train → fit/calibration subsets) with **Isotonic Regression** to calibrate raw classification scores into well-calibrated probabilities, preventing calibration leakage into the validation fold.
  * Executes a **1,000-iteration Bootstrap Analysis** on out-of-fold predictions to calculate empirical confidence intervals for Gini, Brier, and ECE.
  * Generates a 3-step ablation study (Baseline → Optuna-Tuned → Tuned+Calibrated) to quantify the contribution of each modeling stage. Current result:

    | Experiment Step | Gini Index | Δ vs. Baseline |
    | :--- | :--- | :--- |
    | 1. LightGBM Baseline (Default Params) | 0.27975 | +0.00000 |
    | 2. LightGBM Optimized (Optuna Tuning, fold-safe calib. split) | 0.27847 | -0.00128 |
    | 3. LightGBM Final (Tuned + Calibration) | 0.27660 | -0.00315 |

    Note: Optuna's own tuning-time CV Gini (0.28300) is measured without the calibration carve-out. The small apparent dip in step 2 above reflects the reduced training data (~64% vs. ~80% of each outer-training fold) introduced by reserving a calibration subset, not a failure of tuning itself — the trade-off is made deliberately to keep calibration leakage-free.
  * Trains the **final production model** on the complete training dataset using `n_estimators=347` (Optuna's early-stopped average) and saves it for downstream explainability analysis and deployment.
  * Generates the Kaggle submission file and the reliability (calibration) diagram.
  * **Outputs Written To:** `model_results/final_model.pkl`, `model_results/fold_models.pkl`, `model_results/oof_preds_calibrated.npy`, `model_results/test_preds_calibrated.npy`, `model_results/metrics_summary.pkl`, `model_results/fold_results.csv`, `model_results/ablation_study.csv`, `model_results/submission_final.csv`, `model_results/lgbm_reliability_diagram.png`
* **`03b_Shap_Interpretability.py`**
  * Loads the final production model and reproduces its exact feature representation (categorical casting, feature order) to guarantee a faithful explanation.
  * Utilizes **TreeSHAP (SHAP Explainer)** on a 1,000-row random sample to uncover global and local feature behaviors.
  * Generates summary (dot/bar), dependence, force, and waterfall plots, plus a ranked feature-importance table.
  * Top drivers of predicted risk in the current model: `ps_ind_05_cat`, `ps_car_07_cat`, `ps_car_09_cat`, `ps_car_13`, `ps_car_01_cat`.
  * **Outputs Written To:** `model_results/shap_analysis/` (summary plots, dependence plots, force/waterfall plots, `shap_feature_importance.csv`, raw SHAP values)
* **`03c_Business_Impact_Simulation.py`**
  * Translates statistical metrics into financial terms. Simulates expected savings across a sensitivity grid of intervention costs, claim severities, and targeting thresholds (top 0.5%, 1%, 2%).
  * Performs cost-benefit analysis to help actuarial teams identify profitable intervention thresholds under different risk-tolerance assumptions.
  * **Outputs Written To:** `model_results/business_impact_sensitivity.csv`, `model_results/business_impact_heatmaps/`

---

## Technical Insight: Reliability & Calibration

The pipeline achieves a well-calibrated final model, with bootstrap-estimated ECE of 0.00030 (95% CI: [0.00008, 0.00067]) using Isotonic Regression on a fold-safe calibration split. As is typical with isotonic calibration, this comes with a small, expected reduction in raw discrimination (Gini) relative to the uncalibrated model, since isotonic regression's step-function output introduces prediction ties that slightly reduce ranking-based metrics like AUC/Gini while substantially improving probability accuracy.

---

## Project Directory Structure & File Pathways

The pipeline relies on a structured workspace to pass preprocessed data and models between scripts sequentially. Ensure the following paths are maintained:

```text
│── train.csv                         # Original raw training data
│── test.csv                          # Original raw test data
├── eda_results/                      # Generated by the optional Phase 0 EDA script
│   ├── ps_calc_chi2_cramers_v.csv
│   ├── ps_calc_correlation_heatmap.png
│   ├── ps_calc_target_conditioned_distributions.png
│   ├── full_feature_importance_with_ps_calc.csv
│   ├── full_feature_importance_plot.png
│   └── ps_calc_ablation_gini_comparison.csv
├── model_results/                    # Generated automatically across all phases
│   ├── business_impact_heatmaps/
│   ├── shap_analysis/
│   ├── baseline_metrics_backup.pkl
│   ├── best_lgbm_params.pkl
│   ├── final_model.pkl
│   ├── fold_models.pkl
│   ├── ablation_study.csv
│   ├── fold_results.csv
│   ├── submission_final.csv
│   └── lgbm_reliability_diagram.png
├── 00_EDA_ps_calc_justification.py   # Optional — run manually, not part of run_pipeline.py
├── 01_Preprocessing.py
├── 02a_BaselineModels.py
├── 02b_HyperParameterTuning.py
├── 03a_ModelTRaining_Bootstrap.py
├── 03b_Shap_Interpretability.py
├── 03c_Business_Impact_Simulation.py
├── run_pipeline.py
├── X_preprocessed.csv
├── y_preprocessed.csv
├── test_preprocessed.csv
├── metadata.pkl
├── categorical_cols.pkl
├── rare_class_map.pkl
├── README.md
└── model_results/submission_final.csv   # Kaggle submission file
```

*Note: File and folder names above reflect the pipeline's current state, as run via `run_pipeline.py`. The `eda_results/` folder and `00_EDA_ps_calc_justification.py` are produced by the optional Phase 0 script described above and are not created by `run_pipeline.py` itself.*

---

## Getting Started & Execution Order

### 1. Installation
Install all production dependencies:
```bash
pip install pandas numpy lightgbm scikit-learn joblib tqdm shap optuna matplotlib seaborn xgboost scipy
```
*(`scipy` is required for the optional Phase 0 EDA script's chi-square tests.)*

### 2. *(Optional) Feature Removal Justification*
*Run once, manually, before the main pipeline, to reproduce the statistical evidence behind the `ps_calc_*` removal decision (see Key Achievements above):*
```bash
python "00_EDA_ps_calc_justification.py"
```

### 3. Running the Pipeline
Run the full pipeline via **`run_pipeline.py`**, or execute the stages individually in order:
```bash
python 01_Preprocessing.py
python 02a_BaselineModels.py
python 02b_HyperParameterTuning.py
python 03a_ModelTRaining_Bootstrap.py
python 03b_Shap_Interpretability.py
python 03c_Business_Impact_Simulation.py
```

---

## Evaluation Metrics
The pipeline monitors performance across three critical layers:
1.  **Discrimination:** Normalized Gini Coefficient & ROC-AUC.
2.  **Calibration:** Expected Calibration Error (ECE) & Brier Score Loss (ensuring predicted probabilities match actual frequencies).
3.  **Business Utility:** Recall@K% (capturing the maximum amount of high-risk claims within the top 0.5%, 1%, and 2% risk tiers).

4. Model Evolution (Ablation): Baseline → Optuna-Tuned → Calibrated, tracked end-to-end in `model_results/ablation_study.csv` to attribute performance changes to each pipeline stage.

*5. Feature Selection Validity: `ps_calc_*` exclusion is backed by chi-square/Cramér's V tests, correlation structure, importance ranking, and a bootstrap-CI ablation experiment, tracked in `eda_results/`.*