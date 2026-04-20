from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"file://{ROOT / 'mlruns'}"
MODEL_NAME = "wine-quality-model"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def _load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            "No registered model version found. Run Phase 2 notebook first."
        )

    latest = max(versions, key=lambda v: int(v.version))
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest.version}")


def main() -> None:
    st.title("Wine Quality Predictor (Phase 3)")
    st.caption("Using the model from MLflow Model Registry")

    model = _load_model()
    user_values = {}

    with st.form("prediction_form"):
        for feature in FEATURES:
            user_values[feature] = st.slider(feature, min_value=0.0, max_value=15.0, value=1.0, step=0.01)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([user_values], columns=FEATURES)
        prediction = float(model.predict(input_df)[0])

        st.caption(
            "Confidence interval is an illustrative approximation (±0.25) for demo usage."
        )
        lower = prediction - 0.25
        upper = prediction + 0.25
        st.success(
            f"Predicted quality: {prediction:.2f} [95% CI: {lower:.2f} - {upper:.2f}]"
        )


if __name__ == "__main__":
    main()
