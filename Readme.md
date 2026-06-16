# End-to-End Car Insurance Claim Prediction Pipeline with Probability Calibration

An advanced, production-grade machine learning pipeline designed to predict automobile insurance claim probabilities under extreme class imbalance (~3.6% positive rate). Built on top of the Porto Seguro’s Safe Driver Prediction benchmark, this project implements rigorous data preprocessing, leakage-free cross-validation, hyperparameter optimization, probability calibration, bootstrap uncertainty estimation, SHAP-driven interpretability, and operational business impact simulations.

##  Key Achievements & Validation Benchmarks
* **Independent External Validation (Kaggle):** Achieved a **Public Gini Score of 0.27632** and a **Private Gini Score of 0.28394**. The higher private leaderboard score suggests good generalization performance and limited evidence of overfitting.
* **Leakage-Free Reliability:** The internal 5-Fold Stratified Cross-Validation score (**0.27590**) perfectly aligns with the external Kaggle public benchmark, validating the integrity of the evaluation pipeline.
* **Actuarial-Grade Calibration:** Post-calibration Expected Calibration Error (ECE) was minimized to **0.00043** using Isotonic Regression, ensuring predicted probabilities mirror empirical risk frequencies. Brier Score: 0.03477


| Metric | Value |
| :--- | :--- |
| Kaggle Public Gini | 0.27632 |
| Kaggle Private Gini | 0.28394 |
| Internal CV Gini | 0.27641 |
| PR-AUC | 0.06671 |
| Brier Score | 0.03477 |
| Expected Calibration Error (ECE) | 0.00045 |

---

##  Project Architecture & Execution Flow

The pipeline is modularized into dedicated Python scripts, enforcing clean separation of concerns and reproducibility.

###  1. Data Engineering & Pipeline Safety
* **`01_Preprocessing.py`**
  * Handles missing value imputation strategies customized for high-cardinality categorical data and continuous risk indices.
  * Implements strict categorical mapping for rare classes and frequency encoding.
  * Encapsulates all transformations into an un-compromised, fold-isolated framework to structurally prevent data leakage.

###  2. Feature Selection & Baseline Construction
* **`02a_LGBM_FeatureImportance.py`**
  * Trains an initial LightGBM ensemble to compute native split and gain-based feature importances.
  * Conducts non-informative feature pruning to optimize computational efficiency and reduce noise.
* **`02b_ModelBaseline.py`**
  * Establishes the core machine learning baseline using an out-of-the-box LightGBM classifier.
  * Implements 5-Fold Stratified Cross-Validation to benchmark early Gini metrics and log execution times.

###  3. Optimization & Rigorous Evaluation
* **`02c_HyperParameterTuning.py`**
  * Executes a fine-grained, automated hyperparameter sweep (utilizing Optuna/Grid-Search paradigms) optimizing regularizations (`reg_alpha`, `reg_lambda`), tree structures (`num_leaves`, `max_depth`), and learning trajectories.
* **`03a_ModelTRaining_Bootstrap.py`**
  * Trains the optimal, fine-tuned final LightGBM ensemble across the complete stratified fold architecture.
  * Integrates **Isotonic Regression** to calibrate raw classification scores into mathematically sound probabilities.
  * Executes a **1,000-iteration Bootstrap Analysis** to calculate empirical confidence intervals and statistically bound model variance.

###  4. Explainability & Financial Decisioning
* **`03b_Shap_Interpretability.py`**
  * Leverages **TreeSHAP (SHAP Explainer)** to uncover global and local feature behaviors.
  * Generates summary, dependence, and waterfall plots to translate black-box model mechanics into clear, transparent actuarial risk factors.
* **`03c_Business_Impact_Simulation.py`**
  * Performs expected-value based business impact simulations across different intervention costs, claim severities, and targeting thresholds.
  * Performs cost-benefit and sensitivity analyses to establish optimal premium thresholds and measure business profitability under diverse risk-tolerance levels.

---

## Technical Insight: Reliability & Calibration

The pipeline features a state-of-the-art **Expected Calibration Error (ECE) of 0.00043**. 


---

## Project Directory Structure & File Pathways

The pipeline relies on a structured workspace to pass preprocessed data and models between scripts sequentially. Ensure the following paths are maintained:

```text
│── train.csv                         # Original raw training data
│── test.csv                          # Original raw test data
├── model_results/                    # Generated automatically by Phase 1
│   ├── business_impact_heatmaps
│   ├── shap_analysis
├── kaggle submissions/                 # kaggle results
├── 01_Preprocessing.py
├── 02a_LGBM_FeatureImportance.py
├── 02b_ModelBaseline.py
├── 02c_HyperParameterTuning.py
├── 03a_ModelTraining_Bootstrap.py
├── 03b_Shap_Interpretability.py
└── 03c_Business_Impact_Simulation.py
└── --------------
├── x_preprocessed.csv
├── y_preprocessed.csv
├── test_preprocessed
├── metadata.pkl
├── categorical_cols.pkl
├── rare_class_map.pkl
├── ReadMe.md
├── submission_calibrated_final.csv (Kaggle submission file)

```

---

## Pipeline Architecture & Execution Flow

The project is broken down into separate, modular scripts to prevent RAM memory overflow and ensure clean isolation of concerns.

### 1. Data Preprocessing & Artifact Generation
*   **`01_Preprocessing.py`**
    *   **Logic:** Separates features into binary, categorical, and continuous lists. Drops the noisy `ps_calc` columns. Handles rows with missing values and converts variables to Pandas `category` dtype for optimized memory footprint and native LightGBM support.
    *   **Outputs Written To:**
        *   `X_preprocessed.csv` (Features for training)
        *   `y_preprocessed.csv` (Target array)
        *   `test_preprocessed.csv` (Features for deployment/submission)
        *   `metadata.pkl`, `categorical_cols.pkl`, `rare_class_map.pkl` (Metadata & Maps)
    *   *Note: This script copies convenience duplicates of these files directly to the root directory for easy access in later phases.*

### 2. Feature Exploration & Optimization (Baseline)
*   **`02a_LGBM_FeatureImportance.py`**
    *   **Logic:** Trains an initial LightGBM classifier to extract Split/Gain feature importance scores. Helps filter out uninformative features.
*   **`02b_ModelBaseline.py`**
    *   **Logic:** Builds an un-tuned baseline model using vanilla parameters to establish a lower bound performance benchmark.
*   **`02c_HyperParameterTuning.py`**
    *   **Logic:** Uses Optuna / Random Search to systematically tune LightGBM parameters (`num_leaves`, `max_depth`, `learning_rate`, `min_child_samples`).
    *   **Outputs Written To:** `outputs/best_params.pkl`

### 3. Robust Training, Validation, Interpretability & Business Alignment
*   **`03a_ModelTRaining_Bootstrap.py`**
    *   **Logic:** This phase trains the final LightGBM model using the Optuna-optimized hyperparameters obtained in Phase 2. The script performs:
    - 5-Fold Stratified Cross-Validation
    - Leakage-free probability calibration using Isotonic Regression
    - Out-of-fold prediction generation
    - Bootstrap-based confidence interval estimation (1000 iterations)
    - Model evolution and ablation study reporting
    - Final production model training on the full dataset
    - Kaggle submission file generation
    - Reliability diagram generation
    - Persistence of all trained artifacts and evaluation outputs

Calibration is performed using a fold-safe internal calibration split to prevent information leakage between model fitting and probability calibration stages.

The final production model is trained on the complete training dataset using the Optuna-optimized hyperparameters and saved for downstream explainability analysis (SHAP) and deployment.
    *   **Outputs Written To:** `models/lgbm_fold_*.pkl`, `models/isotonic_calibrator_fold_*.pkl`
*   **`03b_Shap_Interpretability.py`**
    *   **Logic:** Utilizes **SHAP (SHapley Additive exPlanations)** TreeExplainer to break open the black-box model. Generates global summary plots and local force plots to show exactly why a driver is flagged as high-risk.
*   **`03c_Business_Impact_Simulation.py`**
    *   **Logic:** Translates statistical metrics into financial terms. Simulates changes in profitability, loss ratios, and retention rates across a spectrum of decision thresholds (Sensitivity Analysis) to help actuarial teams optimize pricing structures.

---

## Getting Started & Execution Order

### 1. Installation
Install all production dependencies:
```bash
pip install pandas numpy lightgbm scikit-learn joblib tqdm shap optuna matplotlib seaborn
```

### 2. Running the Pipeline
Run the scripts sequentially in your terminal:
```bash
python 01_Preprocessing.py
python 02a_LGBM_FeatureImportance.py
python 02b_ModelBaseline.py
python 02c_HyperParameterTuning.py
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
