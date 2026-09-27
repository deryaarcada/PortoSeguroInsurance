# End-to-End Car Insurance Claim Prediction Pipeline with Probability Calibration

An advanced, production-grade machine learning pipeline designed to predict automobile insurance claim probabilities under extreme class imbalance (~3.6% positive rate). Built on top of the Porto Seguro's Safe Driver Prediction benchmark, this project implements rigorous data preprocessing, leakage-free cross-validation, hyperparameter optimization, probability calibration, bootstrap uncertainty estimation, SHAP-driven interpretability, and operational business impact simulations.

## Key Achievements & Validation Benchmarks

* **Independent External Validation (Kaggle):** Achieved a **Public Gini Score of 0.27926** and a **Private Gini Score of 0.28518**. The higher private leaderboard score suggests good generalization performance and limited evidence of overfitting.
* **Leakage-Free Reliability:** The internal 5-Fold Stratified Cross-Validation Gini (0.27763, Bootstrap Mean, 95% CI: [0.27004, 0.28468]) sits close to the external Kaggle public benchmark, validating the integrity of the evaluation pipeline. Every learned preprocessing step — including rare-category collapsing — is fit strictly inside each CV fold's training partition (see `preprocessing_utils.py`), so no validation-fold row influences its own fold's preprocessing.
* **Actuarial-Grade Calibration:** Post-calibration Expected Calibration Error (ECE) was minimized using Isotonic Regression, ensuring predicted probabilities mirror empirical risk frequencies. Bootstrap-estimated ECE: 0.00087 (95% CI: [0.00051, 0.00123]); fold-averaged ECE: 0.00176. Brier Score: 0.03476.
* *__Evidence-Based Feature Reduction:__ The exclusion of the 20 `ps_calc_*` features is supported by a dedicated EDA study (see Phase 0 below), not just heuristic judgment — chi-square tests showed no significant association with the target (all p > 0.05), Cramér's V averaged 0.0004, and a controlled ablation experiment found no statistically detectable Gini difference with vs. without these features (ΔGini = +0.0005, fully overlapping 95% CIs).*

| Metric | Value |
| :--- | :--- |
| Kaggle Public Gini | 0.27926 |
| Kaggle Private Gini | 0.28518 |
| Internal CV Gini (Bootstrap Mean) | 0.27763 (95% CI: 0.27004–0.28468) |
| PR-AUC | 0.06633 |
| Brier Score | 0.03476 |
| Expected Calibration Error (ECE) | 0.00087 (bootstrap) / 0.00176 (fold-averaged) |

*Note: All internal metrics above are computed on out-of-fold (OOF) predictions from the final, leakage-free 5-fold pipeline described below.*

---

## Project Architecture & Execution Flow

The pipeline is modularized into dedicated Python scripts, enforcing clean separation of concerns and reproducibility. The pipeline is orchestrated end-to-end via `run_pipeline.py`, which runs 6 stages in sequence and logs timing/status for each. *An additional, standalone EDA script (Phase 0) is run separately, outside `run_pipeline.py`, to produce evidence for feature-removal decisions.*

*A small shared utility module, `metrics.py`, holds the canonical implementations of metrics that would otherwise be duplicated across scripts — `calculate_ece`, `bootstrap_gini_ci`, `evaluate_ranking`, and `calculate_bootstrap_ci` — and is imported by `00_EDA_ps_calc_justification.py`, `02a_BaselineModels.py`, and `03a_ModelTraining_Bootstrap.py`. It must sit in the same directory as the pipeline scripts (or be on `PYTHONPATH`); it is never run on its own.*

*A second such module, **`preprocessing_utils.py`**, provides `fit_rare_map()` / `apply_rare_map()` — the fold-safe rare-category mapping helpers used by `02a_BaselineModels.py`, `02b_HyperparameterTuning.py`, and `03a_ModelTraining_Bootstrap.py`. **Fold-Safe Preprocessing:** `01_Preprocessing.py` used to learn this mapping (which categorical/ordinal values occur in <1% of rows) once, on the full training set, before any cross-validation split existed — meaning each validation fold's own rows had already helped decide which categories counted as "rare" for that fold. This is not target leakage (no label information is involved), but it is a mild preprocessing leakage that weakens the "fully leakage-free" claim. The mapping is now learned fresh, INSIDE each CV fold, from only that fold's training rows, and applied to that fold's validation rows (and, in `03a`, the test set scored by that fold's model). `01_Preprocessing.py` now only identifies and saves the *candidate* column list (`cols_for_rare.pkl`) — the actual thresholding happens fold-by-fold downstream. The same fit-inside-the-fold pattern should be followed for any future learned preprocessing step (imputation, scaling, target encoding, feature selection, etc.).*

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
  * Casts the candidate rare-category columns to a list (`cols_for_rare.pkl`) but does **not** threshold them here — see the "Fold-Safe Preprocessing" note below.
  * Outputs `X_preprocessed.csv`, `y_preprocessed.csv`, `test_preprocessed.csv`, `metadata.pkl`, `categorical_cols.pkl`, `cols_for_rare.pkl`, and copies convenience duplicates to the root directory for later phases.

### 2. Baseline Construction & Optimization
* **`02a_BaselineModels.py`**
  * Benchmarks LightGBM, Random Forest, and XGBoost out-of-the-box, using ALL preprocessed features (no upfront feature-selection step), via 5-Fold Stratified Cross-Validation.
  * This "all-features" design deliberately avoids a feature-selection-leakage pitfall that would occur if a Top-N feature list were derived from the full dataset and then reused across CV folds.
  * Reports Gini, PR-AUC, and Recall@1% per fold plus a Bootstrap 95% CI per model.
  * Current baseline result: LightGBM Gini 0.27946 [0.2714–0.2866], Random Forest 0.24895 [0.2413–0.2556], XGBoost 0.27550 [0.2682–0.2833].
  * Saves `model_results/baseline_metrics_backup.pkl` for later use in the ablation study.
  * (A separate, optional descriptive-only feature-importance script exists for exploratory reporting but is not part of the active `run_pipeline.py` sequence, since its output is not used to filter or select features downstream.)
* **`02b_HyperparameterTuning.py`**
  * Executes a 30-trial Bayesian hyperparameter search via Optuna (TPE sampler), optimizing `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, and `learning_rate`, using 5-Fold Stratified CV with early stopping.
  * Current best result: CV Gini 0.28329, with an average early-stopped tree count of 364 (search upper bound: n_estimators=1200).
  * Also records `n_estimators_effective` — the average early-stopped tree count for the best trial — so Phase 3a can train the final production model with a tree count consistent with the reported tuning Gini, instead of the (larger) search-space upper bound.
  * **Outputs Written To:** `model_results/best_lgbm_params.pkl`

### 3. Robust Training, Validation, Interpretability & Business Alignment
* **`03a_ModelTraining_Bootstrap.py`**
  * Trains the tuned LightGBM model across 5 Stratified folds using the Optuna-optimized hyperparameters.
  * Integrates a **fold-safe 80/20 calibration split** (train → fit/calibration subsets) with **Isotonic Regression** to calibrate raw classification scores into well-calibrated probabilities, preventing calibration leakage into the validation fold.
  * Executes a **1,000-iteration Bootstrap Analysis** on out-of-fold predictions to calculate empirical confidence intervals for Gini, Brier, and ECE.
  * Generates a 3-step ablation study (Baseline → Optuna-Tuned → Tuned+Calibrated) to quantify the contribution of each modeling stage. Current result:

    | Experiment Step | Gini Index | Δ vs. Baseline |
    | :--- | :--- | :--- |
    | 1. LightGBM Baseline (Default Params) | 0.27946 | +0.00000 |
    | 2. LightGBM Optimized (Optuna Tuning, fold-safe calib. split) | 0.27904 | -0.00042 |
    | 3. LightGBM Final (Tuned + Calibration) | 0.27740 | -0.00206 |

    Note: Optuna's own tuning-time CV Gini (0.28329) is measured without the calibration carve-out. The small apparent dip in step 2 above reflects the reduced training data (~64% vs. ~80% of each outer-training fold) introduced by reserving a calibration subset, not a failure of tuning itself — the trade-off is made deliberately to keep calibration leakage-free.
  * Trains the **final production model** on the complete training dataset using `n_estimators=364` (Optuna's early-stopped average) and saves it for downstream explainability analysis and deployment.
  * Generates the Kaggle submission file and the reliability (calibration) diagram.
  * Saves **`model_results/calibrated_predictor.pkl`** — a self-contained bundle (`{fold_models, feature_names, categorical_features}`, where each entry in `fold_models` carries its own `{model, iso_reg, rare_map}`) that is the actual predictor behind `oof_preds_calibrated.npy` and `submission_final.csv`: score new data by averaging each fold model's `predict_proba()` through that fold's own isotonic calibrator, after applying that fold's own rare-category map (see `predict_calibrated_ensemble()` in the script). This exists because `final_model.pkl` (below) is a *single, uncalibrated* LightGBM model kept only so `03b_SHAP_Interpretability.py` has a plain tree model for TreeSHAP — its `.predict_proba()` does **not** reproduce the calibrated numbers reported elsewhere in this README. **Use `calibrated_predictor.pkl` for scoring, validation reproduction, or audits; use `final_model.pkl` only for SHAP/interpretability.**
  * **Outputs Written To:** `model_results/final_model.pkl`, `model_results/calibrated_predictor.pkl`, `model_results/fold_models.pkl`, `model_results/oof_preds_calibrated.npy`, `model_results/test_preds_calibrated.npy`, `model_results/metrics_summary.pkl`, `model_results/fold_results.csv`, `model_results/ablation_study.csv`, `model_results/submission_final.csv`, `model_results/lgbm_reliability_diagram.png`
* **`03b_SHAP_Interpretability.py`**
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

The pipeline achieves a well-calibrated final model, with bootstrap-estimated ECE of 0.00087 (95% CI: [0.00051, 0.00123]) using Isotonic Regression on a fold-safe calibration split. As is typical with isotonic calibration, this comes with a small, expected reduction in raw discrimination (Gini) relative to the uncalibrated model, since isotonic regression's step-function output introduces prediction ties that slightly reduce ranking-based metrics like AUC/Gini while substantially improving probability accuracy.

*Note: the ECE figures above reflect two corrections made to the pipeline. First, a fix to `calculate_ece()`: the metric previously mixed quantile-based bins (for the accuracy/confidence values) with equal-width bin weights, which understated ECE by roughly 2.5–4x; the corrected implementation bins consistently by quantile throughout and now lives in a single shared module, `metrics.py`, imported by every script that needs it (see directory structure below). Second, the fold-safe rare-category mapping fix described below, which slightly changes every downstream number (Gini, ECE, hyperparameters) since each fold's training data is no longer identical to what it was before. The model's calibration quality itself remains excellent — an ECE around 0.0009–0.0018 is well below typical "well-calibrated" thresholds of ~0.05.*

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
│   ├── final_model.pkl              # Raw LightGBM, no calibration — SHAP/interpretability use only
│   ├── calibrated_predictor.pkl     # THE actual predictor (fold ensemble + isotonic calibrators);
│   │                                 #   use this for scoring, validation reproduction, or audits
│   ├── fold_models.pkl
│   ├── ablation_study.csv
│   ├── fold_results.csv
│   ├── submission_final.csv
│   └── lgbm_reliability_diagram.png
├── metrics.py                         # Shared metrics module (ECE, bootstrap Gini CI, ranking
│                                       #   metrics); imported by 00, 02a, and 03a — not run standalone
├── preprocessing_utils.py             # Shared fold-safe preprocessing helpers (fit_rare_map /
│                                       #   apply_rare_map); imported by 02a, 02b, and 03a — not run standalone
├── 00_EDA_ps_calc_justification.py   # Optional — run manually, not part of run_pipeline.py
├── 01_Preprocessing.py
├── 02a_BaselineModels.py
├── 02b_HyperparameterTuning.py
├── 03a_ModelTraining_Bootstrap.py
├── 03b_SHAP_Interpretability.py
├── 03c_Business_Impact_Simulation.py
├── run_pipeline.py
├── requirements.txt                   # Pinned production dependencies — see "Installation" below
├── requirements-dev.txt               # Dev-only deps (pytest) needed to run tests/, not the pipeline
├── tests/
│   └── test_pipeline_basics.py        # Small pytest sanity checks — see "Testing" below
├── X_preprocessed.csv
├── y_preprocessed.csv
├── test_preprocessed.csv
├── metadata.pkl
├── categorical_cols.pkl
├── cols_for_rare.pkl                  # Candidate rare-category columns only (names, not thresholds);
│                                       #   actual rare-value thresholding happens per CV fold downstream
├── README.md
└── model_results/submission_final.csv   # Kaggle submission file
```

*Note: File and folder names above reflect the pipeline's current state, as run via `run_pipeline.py`. The `eda_results/` folder and `00_EDA_ps_calc_justification.py` are produced by the optional Phase 0 script described above and are not created by `run_pipeline.py` itself.*

---

## Getting Started & Execution Order

### 0. Python Version
This project targets **Python 3.12** (developed and tested on 3.12.10). This isn't an arbitrary choice: the pinned versions in `requirements.txt` (e.g. `numpy==1.26.4`) predate Python 3.14 and have no pre-built wheel for it, so `pip install` on 3.14 fails or falls back to a source build that typically fails without a C/C++ compiler. If you're on a newer Python (3.13+), either install Python 3.12 alongside it and create a virtual environment with it (`py -3.12 -m venv venv` on Windows, then `.\venv\Scripts\activate`), or update the pins in `requirements.txt` to versions with wheels for your Python version and re-verify the pipeline still reproduces the metrics reported here.

### 1. Installation
Install pinned production dependencies from `requirements.txt` (Python 3.12 environment — see above):
```bash
pip install -r requirements.txt
```
Versions are pinned deliberately, not left as a loose `pip install pandas numpy ...` list: this pipeline's core outputs are pickled models (`final_model.pkl`, `calibrated_predictor.pkl`, `fold_models.pkl`) and reported metrics (Gini, ECE, Brier) whose exact values — and in the case of pickles, whose very ability to load — depend on the library versions used to produce them. scikit-learn, LightGBM, XGBoost, and SHAP have all changed default behavior or pickle compatibility across versions before. If you deliberately upgrade a pinned version, re-run the pipeline and confirm the metrics in this README still hold.

To also run the test suite (see "Testing" below), additionally install:
```bash
pip install -r requirements-dev.txt
```

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
python 02b_HyperparameterTuning.py
python 03a_ModelTraining_Bootstrap.py
python 03b_SHAP_Interpretability.py
python 03c_Business_Impact_Simulation.py
```

### 4. Testing (optional, but recommended after any code change)
A small `pytest` suite in `tests/` covers a handful of fast, high-value sanity checks — it is not a substitute for re-running the full pipeline, but it catches obvious regressions in seconds instead of the ~50 minutes a full run takes:
```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```
What it checks:
- **Preprocessing consistency:** `X_preprocessed.csv` and `test_preprocessed.csv` have identical columns, in identical order.
- **Prediction validity:** every value in `test_preds_calibrated.npy` / `oof_preds_calibrated.npy` is a finite number in `[0, 1]`.
- **`calculate_ece()` correctness:** returns exactly 0 for a toy case of perfect predictions, and a large value for a toy case of confidently wrong ones (the second check guards specifically against the quantile/equal-width bin-mismatch bug described earlier in this README).
- **`load_predictions()` error handling:** raises a clear `FileNotFoundError` — naming the missing path and pointing at Phase 3a — rather than letting a confusing low-level error surface instead.

The two artifact-based checks (preprocessing columns, prediction validity) `pytest.skip()` with an explanatory message if the relevant pipeline stage hasn't been run yet, rather than failing — they check the pipeline's actual output, not a reimplementation of its logic, so they're only meaningful once real files exist to check.

---

## Evaluation Metrics
The pipeline monitors performance across three critical layers:
1.  **Discrimination:** Normalized Gini Coefficient & ROC-AUC.
2.  **Calibration:** Expected Calibration Error (ECE) & Brier Score Loss (ensuring predicted probabilities match actual frequencies).
3.  **Business Utility:** Recall@K% (capturing the maximum amount of high-risk claims within the top 0.5%, 1%, and 2% risk tiers).

4. Model Evolution (Ablation): Baseline → Optuna-Tuned → Calibrated, tracked end-to-end in `model_results/ablation_study.csv` to attribute performance changes to each pipeline stage.

*5. Feature Selection Validity: `ps_calc_*` exclusion is backed by chi-square/Cramér's V tests, correlation structure, importance ranking, and a bootstrap-CI ablation experiment, tracked in `eda_results/`.*