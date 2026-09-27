"""
metrics.py
================================================================
Standart, shared metrics helpers for the claim-prediction pipeline.

WHY THIS FILE EXISTS
---------------------
Several metric functions (calculate_ece, bootstrap_gini_ci, evaluate_ranking)
were previously copy-pasted across multiple scripts. Copy-pasted code drifts: the ECE
function in 03a_ModelTraining_Bootstrap.py had silently diverged from its
notebook origin (quantile bins used for the accuracy/confidence values, but
equal-width bins used for the bin weights), producing a subtly wrong ECE.

To prevent that from happening again, every script and notebook in this
project should import its metric functions from HERE instead of redefining
them locally. This module is the single source of truth.

USAGE
-----
    from metrics import calculate_ece, bootstrap_gini_ci, evaluate_ranking, calculate_bootstrap_ci

Keep this file in the same directory as the pipeline scripts (or on the
PYTHONPATH) so the imports resolve without changes to sys.path.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc as sklearn_auc
from sklearn.utils import resample

__all__ = [
    "calculate_ece",
    "bootstrap_gini_ci",
    "evaluate_ranking",
    "calculate_bootstrap_ci",
]


# ============================================================================
# CALIBRATION
# ============================================================================

def calculate_ece(y_true, y_probs, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) using quantile (equal-count) bins.

    Predictions are sorted and split into `n_bins` equal-count chunks (the
    last chunk absorbs any remainder from array_split). For each chunk, ECE
    accumulates |mean(y_true) - mean(y_probs)| weighted by the chunk's share
    of the total sample. Binning is used consistently for both the
    accuracy/confidence values AND the weights, unlike an earlier version of
    this function that mixed quantile bins for one and equal-width bins for
    the other.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels (0/1).
    y_probs : array-like of shape (n_samples,)
        Predicted probabilities.
    n_bins : int, default=10
        Number of quantile bins.

    Returns
    -------
    float
        The Expected Calibration Error.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    order = np.argsort(y_probs)
    chunks = np.array_split(order, n_bins)

    n_total = len(y_true)
    ece = 0.0
    for idx in chunks:
        if len(idx) == 0:
            continue
        acc = y_true[idx].mean()
        conf = y_probs[idx].mean()
        ece += (len(idx) / n_total) * abs(acc - conf)

    return ece


# ============================================================================
# DISCRIMINATION (GINI) — BOOTSTRAP CI
# ============================================================================

def bootstrap_gini_ci(y_true, y_probs, n_iterations=200):
    """
    Bootstrap a 95% confidence interval for the Gini coefficient (2*AUC - 1).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_probs : array-like of shape (n_samples,)
    n_iterations : int, default=200
        Number of bootstrap resamples.

    Returns
    -------
    (mean, lower, upper) : tuple of float
        Bootstrap mean Gini and the 2.5th / 97.5th percentile bounds.
    """
    stats = []
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    for i in range(n_iterations):
        y_true_resample, y_probs_resample = resample(y_true, y_probs, random_state=i)
        auc_boot = roc_auc_score(y_true_resample, y_probs_resample)
        stats.append(2 * auc_boot - 1)

    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return np.mean(stats), lower, upper


def calculate_bootstrap_ci(y_true, y_probs, n_bootstraps=1000, show_progress=False):
    """
    Bootstrap 95% confidence intervals for Gini, Brier Score, and ECE together,
    resampling the same indices for all three metrics on each iteration.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_probs : array-like of shape (n_samples,)
    n_bootstraps : int, default=1000
    show_progress : bool, default=False
        If True and tqdm is available, shows a progress bar.

    Returns
    -------
    dict
        {'Gini Index': (lower, mean, upper),
         'Brier Score': (lower, mean, upper),
         'ECE': (lower, mean, upper)}
    """
    y_true_arr = np.asarray(y_true)
    y_probs_arr = np.asarray(y_probs)

    bootstrap_gini, bootstrap_brier, bootstrap_ece = [], [], []

    iterator = range(n_bootstraps)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator)
        except ImportError:
            pass

    for i in iterator:
        indices = resample(np.arange(len(y_true_arr)), replace=True, random_state=i)
        y_true_sample = y_true_arr[indices]
        y_probs_sample = y_probs_arr[indices]

        sample_auc = roc_auc_score(y_true_sample, y_probs_sample)
        bootstrap_gini.append(2 * sample_auc - 1)
        bootstrap_brier.append(brier_score_loss(y_true_sample, y_probs_sample))
        bootstrap_ece.append(calculate_ece(y_true_sample, y_probs_sample))

    metrics = {
        'Gini Index': bootstrap_gini,
        'Brier Score': bootstrap_brier,
        'ECE': bootstrap_ece
    }

    ci_results = {}
    for name, values in metrics.items():
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        mean = np.mean(values)
        ci_results[name] = (lower, mean, upper)

    return ci_results


# ============================================================================
# RANKING / BUSINESS METRICS
# ============================================================================

def evaluate_ranking(y_true, y_probs, k_percent_list=(0.005, 0.01, 0.02)):
    """
    Calculate PR-AUC and Recall@K% metrics.

    Key format note: keys are named f'Recall@{int(k * 100)}%' (e.g. k=0.01 ->
    'Recall@1%'). An earlier duplicate of this function omitted the int()
    cast, which for some values of k produced keys like 'Recall@1.0%' instead
    of 'Recall@1%', a silent mismatch that forced a defensive "search for
    any key containing 'Recall@1'" workaround downstream. Using this shared,
    consistently-formatted function removes the need for that workaround.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_probs : array-like of shape (n_samples,)
    k_percent_list : iterable of float, default=(0.005, 0.01, 0.02)
        Top-k percentiles (as fractions) at which to compute recall.

    Returns
    -------
    dict
        {'PR-AUC': ..., 'Recall@0%': ..., 'Recall@1%': ..., ...}
    """
    results = {}

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    results['PR-AUC'] = sklearn_auc(recall, precision)

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_probs = pd.Series(y_probs).reset_index(drop=True)
    df_temp = pd.DataFrame({'y_true': y_true, 'y_probs': y_probs}).sort_values(by='y_probs', ascending=False)
    n_total_positives = y_true.sum()

    for k in k_percent_list:
        n_cutoff = max(1, int(len(y_true) * k))
        n_positives_at_k = df_temp.iloc[:n_cutoff]['y_true'].sum()
        recall_at_k = n_positives_at_k / n_total_positives if n_total_positives > 0 else 0
        results[f'Recall@{int(k * 100)}%'] = recall_at_k

    return results
