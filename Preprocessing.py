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

    # ordinal vs real continuous (cardinality based)
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

    # ---- continuous: missing flag + median impute
    for col in real_cont_cols:
        if (df[col] == -1).any():
            df[col + "_missing"] = (df[col] == -1).astype(int)
            median_val = df.loc[df[col] != -1, col].median()
            df[col] = df[col].replace(-1, median_val)

    # ---- ordinal: rare class collapse (<1%)
    for col in ordinal_cols:
        freq = df[col].value_counts(normalize=True)
        rare_classes = freq[freq < 0.01].index
        df[col] = df[col].replace(rare_classes, -1)

    # ---- label encoding
    for col in ordinal_cols + categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # ---- final feature set
    tree_features = (
        ordinal_cols +
        real_cont_cols +
        binary_cols +
        categorical_cols +
        [c for c in df.columns if c.endswith("_missing")]
    )

    X = df[tree_features]
    y = df["target"]

    return X, y, tree_features
