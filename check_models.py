
from pathlib import Path
import joblib
import traceback

model_dir = Path("saved_models")

files = [
    "xgboost_churn_model.pkl",
    "churn_preprocessor.pkl",
    "cluster_preprocessor.pkl",
    "kmeans_model.pkl",
]

for filename in files:
    path = model_dir / filename
    print(f"\nChecking: {filename}")

    if not path.exists():
        print("❌ FILE NOT FOUND")
        continue

    print(f"File size: {path.stat().st_size:,} bytes")

    try:
        obj = joblib.load(path)
        print(f"✅ Loaded successfully: {type(obj).__name__}")
    except Exception:
        print("❌ FAILED TO LOAD")
        traceback.print_exc()
