# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------
# FEATURE GROUPS
# -------------------------------------------------
def get_feature_groups(df):

    binary_cols = [c for c in df.columns if c.endswith("_bin")]
    categorical_cols = [c for c in df.columns if c.endswith("_cat")]

    continuous_cols = [
        c for c in df.columns
        if c not in binary_cols + categorical_cols + ["id", "target"]
    ]

    cardinality = df[continuous_cols].nunique()
    ordinal_cols = cardinality[cardinality <= 30].index.tolist()
    real_cont_cols = cardinality[cardinality > 30].index.tolist()

    return binary_cols, categorical_cols, ordinal_cols, real_cont_cols


# -------------------------------------------------
# MAIN PREPROCESS FUNCTION
# -------------------------------------------------
def preprocess(path):

    df = pd.read_csv(path)

    # ---- drop id & ps_calc
    df.drop(columns=["id"], inplace=True)
    ps_calc_cols = [c for c in df.columns if c.startswith("ps_calc")]
    df.drop(columns=ps_calc_cols, inplace=True)

    # ---- feature groups
    binary_cols, categorical_cols, ordinal_cols, real_cont_cols = get_feature_groups(df)

    # -------------------------------------------------
    # CATBOOST CATEGORICAL SELECTION
    # -------------------------------------------------
    cat_cols_cb = []
    for col in categorical_cols:
        nunique = df[col].nunique()
        missing = (df[col] == -1).mean()

        if nunique > 2 and nunique <= 100 and missing < 0.8:
            cat_cols_cb.append(col)

    # remaining categorical → numeric
    cat_cols_numeric = list(set(categorical_cols) - set(cat_cols_cb))

    # -------------------------------------------------
    # CONTINUOUS
    # -------------------------------------------------
    for col in real_cont_cols:
        if (df[col] == -1).any():
            df[col + "_missing"] = (df[col] == -1).astype(int)
            median_val = df.loc[df[col] != -1, col].median()
            df[col] = df[col].replace(-1, median_val)

    # -------------------------------------------------
    # ORDINAL RARE COLLAPSE
    # -------------------------------------------------
    for col in ordinal_cols:
        freq = df[col].value_counts(normalize=True)
        rare_classes = freq[freq < 0.01].index
        df[col] = df[col].replace(rare_classes, -1)

    # -------------------------------------------------
    # LABEL ENCODE (ONLY NUMERIC ONES)
    # -------------------------------------------------
    for col in ordinal_cols + cat_cols_numeric:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # -------------------------------------------------
    # FINAL FEATURE SET
    # -------------------------------------------------
    tree_features = (
        ordinal_cols +
        real_cont_cols +
        binary_cols +
        cat_cols_cb +
        cat_cols_numeric +
        [c for c in df.columns if c.endswith("_missing")]
    )

    X = df[tree_features]
    y = df["target"]

    # CatBoost wants indices
    cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols_cb]

    return X, y, tree_features, cat_cols_cb, cat_feature_indices
