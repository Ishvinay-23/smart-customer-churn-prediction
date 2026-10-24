from pathlib import Path
import xgboost as xgb

model_path = Path("saved_models/xgboost_churn_model.json")

print("File exists:", model_path.exists())
print("File size:", model_path.stat().st_size if model_path.exists() else 0)

model = xgb.XGBClassifier()
model.load_model(str(model_path))

print("Model loaded successfully!")
print("Model type:", type(model).__name__)
print("Number of features:", model.n_features_in_)