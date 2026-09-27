"""
tests/test_pipeline_basics.py
================================================================
Small, fast sanity checks for the pipeline. These are NOT a substitute
for running the full pipeline against real data — the goal is to catch
obvious regressions quickly (a broken import, a metric function returning
nonsense, a missing-file error message disappearing) without needing the
595k-row Porto Seguro dataset to be present.

Run from the project root, with requirements.txt and requirements-dev.txt
installed:

    pip install -r requirements.txt -r requirements-dev.txt
    pytest

Two kinds of tests here:
  - Pure unit tests (calculate_ece on a toy case, load_predictions'
    error handling) that always run, regardless of what data is present.
  - Artifact checks (preprocessing output columns, prediction ranges)
    that run against the ACTUAL files this pipeline produces
    (X_preprocessed.csv, test_preprocessed.csv, test_preds_calibrated.npy,
    oof_preds_calibrated.npy) if they exist, and are skipped with a clear
    message otherwise. These intentionally check the real pipeline's
    output rather than a reimplementation of its logic, so they only pass
    if 01_Preprocessing.py / 03a_ModelTraining_Bootstrap.py actually
    behaved correctly when you last ran them.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from metrics import calculate_ece  # noqa: E402


def _load_module_from_file(module_name: str, filename: str):
    """
    Import a pipeline script whose filename starts with a digit (e.g.
    '03c_Business_Impact_Simulation.py'), which a plain `import
    module_name` cannot do since that isn't a valid Python identifier.
    """
    file_path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(file_path):
        pytest.skip(f"{filename} not found in project root — cannot test it.")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# 1. Preprocessing output columns: train and test must match
# ============================================================================

def test_preprocessing_train_test_columns_match():
    """
    X_preprocessed.csv and test_preprocessed.csv must have identical
    columns, in identical order — every downstream script (02a, 02b, 03a)
    assumes this when it applies a fold's model/rare-map to both.
    """
    x_path = os.path.join(PROJECT_ROOT, "X_preprocessed.csv")
    test_path = os.path.join(PROJECT_ROOT, "test_preprocessed.csv")

    if not (os.path.exists(x_path) and os.path.exists(test_path)):
        pytest.skip(
            "X_preprocessed.csv / test_preprocessed.csv not found — "
            "run 01_Preprocessing.py first to generate them."
        )

    # nrows=0 reads only the header — fast even on the full 595k-row file.
    train_cols = list(pd.read_csv(x_path, nrows=0).columns)
    test_cols = list(pd.read_csv(test_path, nrows=0).columns)

    assert train_cols == test_cols, (
        "Train and test preprocessed columns differ.\n"
        f"Train-only: {sorted(set(train_cols) - set(test_cols))}\n"
        f"Test-only: {sorted(set(test_cols) - set(train_cols))}\n"
        f"(Column order matters too, even if the sets above are both empty.)"
    )


# ============================================================================
# 2. Prediction arrays must be valid probabilities
# ============================================================================

def test_predictions_are_valid_probabilities():
    """
    test_preds_calibrated.npy / oof_preds_calibrated.npy must contain
    finite values in [0, 1] — anything outside that range means the
    isotonic calibration step (or something upstream of it) produced
    garbage.
    """
    candidate_files = [
        "model_results/test_preds_calibrated.npy",
        "model_results/oof_preds_calibrated.npy",
    ]
    checked_any = False

    for filename in candidate_files:
        path = os.path.join(PROJECT_ROOT, filename)
        if not os.path.exists(path):
            continue
        checked_any = True
        preds = np.load(path)

        assert preds.size > 0, f"{filename} is empty."
        assert np.all(np.isfinite(preds)), f"{filename} contains NaN/inf values."
        assert preds.min() >= 0.0, f"{filename} has a prediction below 0: {preds.min()}"
        assert preds.max() <= 1.0, f"{filename} has a prediction above 1: {preds.max()}"

    if not checked_any:
        pytest.skip(
            "No prediction files found under model_results/ — "
            "run 03a_ModelTraining_Bootstrap.py first."
        )


# ============================================================================
# 3. calculate_ece() sanity checks (pure unit tests, no data files needed)
# ============================================================================

def test_calculate_ece_perfect_predictions_is_zero():
    """
    If predicted probabilities exactly match the binary outcomes (perfect
    calibration), ECE must be exactly 0 — every quantile bin's mean
    prediction equals its mean outcome by construction.
    """
    rng = np.random.RandomState(0)
    y_true = rng.binomial(1, 0.5, size=500).astype(float)
    y_probs = y_true.copy()  # perfect predictions

    ece = calculate_ece(y_true, y_probs, n_bins=10)

    assert ece == pytest.approx(0.0, abs=1e-12), (
        f"Expected ECE=0 for perfect predictions, got {ece}"
    )


def test_calculate_ece_detects_miscalibration():
    """
    The flip side of the above: a badly miscalibrated toy case (confident
    low predictions on a set that is actually 50% positive) should NOT
    report an ECE near 0. This guards against a future refactor of
    calculate_ece() that accidentally returns ~0 for everything (which is
    exactly the kind of bug that motivated fixing this function in the
    first place).
    """
    y_true = np.array([1.0] * 50 + [0.0] * 50)
    y_probs = np.full(100, 0.05)  # confidently wrong for the 50 positives

    ece = calculate_ece(y_true, y_probs, n_bins=10)

    assert ece > 0.3, (
        f"Expected a large ECE for this badly miscalibrated toy case, got {ece}"
    )


# ============================================================================
# 4. load_predictions() must raise a clear error when the file is missing
# ============================================================================

def test_load_predictions_missing_file_raises_clear_error(tmp_path):
    """
    03c_Business_Impact_Simulation.load_predictions() should fail loudly
    and clearly (a FileNotFoundError that names the missing path and
    points at Phase 3a) rather than letting a confusing low-level numpy
    error surface instead.
    """
    module = _load_module_from_file(
        "biz_impact_module_under_test", "03c_Business_Impact_Simulation.py"
    )

    missing_path = str(tmp_path / "does_not_exist.npy")

    with pytest.raises(FileNotFoundError) as exc_info:
        module.load_predictions(missing_path)

    message = str(exc_info.value)
    assert missing_path in message, f"Error message should name the missing path: {message}"
    assert "Phase 3a" in message, f"Error message should point at Phase 3a: {message}"
