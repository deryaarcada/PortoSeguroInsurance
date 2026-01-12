from preprocessing import preprocess
from modeling import run_lgbm_cv

X, y, features = preprocess("train.csv")
run_lgbm_cv(X, y)