'''
rom Preprocessing import preprocess
from Modeling import run_lgbm_cv

X, y, features = preprocess("train.csv")
run_lgbm_cv(X, y)
'''

# run.py
from Preprocessing import preprocess
from Modeling import run_cv

X, y, tree_features, cat_cols_cb, cat_feature_indices = preprocess("train.csv")

run_cv(
    X,
    y,
    cat_feature_indices,
    n_splits=5
)
