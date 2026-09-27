"""
preprocessing_utils.py
================================================================
Shared, fold-safe preprocessing helpers.

WHY THIS FILE EXISTS
---------------------
The rare-category mapping (collapsing categorical/ordinal values that occur
in <1% of rows into a sentinel value of -1) used to be learned ONCE on the
full training set inside 01_Preprocessing.py, before any cross-validation
split happened. That is not target leakage (no label information is used),
but it does mean each validation fold's rows helped decide which categories
count as "rare" for the very fold they are held out from — a mild form of
preprocessing leakage.

fit_rare_map() / apply_rare_map() below let every CV loop (in
02a_BaselineModels.py, 02b_HyperparameterTuning.py, and
03a_ModelTraining_Bootstrap.py) learn the rare-category mapping from ONLY
that fold's training partition, then apply it to that fold's validation
partition (and, where relevant, the held-out test set scored by that fold's
model). This is also the pattern to follow for any future preprocessing
step that is *learned* from data (imputation values, scalers, target
encoders, feature selection, etc.): fit on the fold's train split only,
apply to everything else.

USAGE
-----
    from preprocessing_utils import fit_rare_map, apply_rare_map

    fold_rare_map = fit_rare_map(X_train, cols_for_rare, threshold=0.01)
    X_train = apply_rare_map(X_train, fold_rare_map)
    X_val = apply_rare_map(X_val, fold_rare_map)

Keep this file in the same directory as the pipeline scripts (or on the
PYTHONPATH) so the imports resolve without changes to sys.path.
"""

__all__ = ["fit_rare_map", "apply_rare_map"]


def fit_rare_map(X_train, cols, threshold=0.01):
    """
    Learn a rare-category mapping from a TRAINING partition only.

    For each column in `cols`, any category whose frequency in `X_train`
    falls below `threshold` is recorded as "rare". Nothing outside
    `X_train` is looked at, so this is safe to call once per CV fold on
    that fold's training split.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training partition (e.g. a single CV fold's training rows).
    cols : iterable of str
        Candidate categorical/ordinal column names to check. Columns not
        present in `X_train` are silently skipped.
    threshold : float, default=0.01
        Categories with frequency below this threshold are marked rare.

    Returns
    -------
    dict
        {column_name: [rare_value, rare_value, ...]}. Only columns that
        actually have at least one rare value are included.
    """
    rare_map = {}
    for col in cols:
        if col not in X_train.columns:
            continue
        try:
            freqs = X_train[col].value_counts(normalize=True)
            rare_values = freqs[freqs < threshold].index.tolist()
            if len(rare_values) > 0:
                rare_map[col] = rare_values
        except Exception:
            # Skip columns that cannot be processed (mirrors the original
            # preprocessing script's defensive behavior).
            continue
    return rare_map


def apply_rare_map(X_data, rare_map):
    """
    Apply a previously-fit rare-category mapping to any DataFrame
    (validation fold, test set, or new data at inference time).

    Parameters
    ----------
    X_data : pd.DataFrame
        Data to transform. Not mutated in place — a copy is returned.
    rare_map : dict
        As returned by fit_rare_map(): {column_name: [rare_value, ...]}.

    Returns
    -------
    pd.DataFrame
        A copy of `X_data` with rare values replaced by -1 in every column
        present in both `X_data` and `rare_map`.
    """
    X_data = X_data.copy()
    for col, rare_values in rare_map.items():
        if col in X_data.columns:
            X_data[col] = X_data[col].replace(rare_values, -1)
    return X_data
