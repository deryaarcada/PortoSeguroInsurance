import joblib

bundle = joblib.load("model_results/calibrated_predictor.pkl")

counts = [len(fm["rare_map"]) for fm in bundle["fold_models"]]
print("Columns affected per fold:", counts)
print("Min / Max:", min(counts), "/", max(counts))

# Optional: effective cardinality of ps_car_11_cat in each fold
for fm in bundle["fold_models"]:
    rare = fm["rare_map"].get("ps_car_11_cat", [])
    print(f"Fold {fm['fold']}: {len(rare)} rare categories merged in ps_car_11_cat "
          f"(about {104 - len(rare) + 1} levels remain)")