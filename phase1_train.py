from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "winequality-red.csv"
MODEL_PATH = ROOT / "docs" / "model.json"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    feature_names = [col for col in df.columns if col not in {"quality", "Id"}]
    X = df[feature_names]
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    payload = {
        "model_type": "Ridge",
        "target": "quality",
        "feature_names": feature_names,
        "intercept": float(model.intercept_),
        "coefficients": [float(c) for c in model.coef_],
        "metrics": metrics,
        "notes": "This model is exported for browser-side inference via JS.",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved browser model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
