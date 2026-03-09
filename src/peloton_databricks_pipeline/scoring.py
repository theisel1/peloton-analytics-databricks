from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ride_duration",
    "distance",
    "calories",
    "avg_cadence",
    "avg_heart_rate",
    "avg_resistance",
    "avg_speed",
]
DISCIPLINE_COLUMN = "fitness_discipline"


def _normalize_model_path(model_path: str) -> str:
    if model_path.startswith("dbfs:/"):
        return model_path.replace("dbfs:/", "/dbfs/", 1)
    return model_path


def load_model_bundle(model_path: str) -> dict[str, Any]:
    resolved = _normalize_model_path(model_path)
    path = Path(resolved)
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if not exists:
        raise FileNotFoundError(f"Model artifact was not found: {resolved}")

    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Model artifact must be a dictionary bundle.")
    return bundle


def _prepare_features(features_df: pd.DataFrame) -> pd.DataFrame:
    frame = features_df.copy()
    if DISCIPLINE_COLUMN not in frame.columns:
        frame[DISCIPLINE_COLUMN] = "unknown"

    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame[DISCIPLINE_COLUMN] = frame[DISCIPLINE_COLUMN].fillna("unknown").astype(str).str.strip().str.lower()
    return frame[[*FEATURE_COLUMNS, DISCIPLINE_COLUMN]]


def predict_total_work(
    features_df: pd.DataFrame,
    model_bundle: dict[str, Any],
) -> pd.DataFrame:
    features = _prepare_features(features_df)

    if "stage1_classifier" in model_bundle and "stage2_regressor" in model_bundle:
        stage1 = model_bundle["stage1_classifier"]
        stage2 = model_bundle["stage2_regressor"]
        threshold = float(model_bundle.get("stage1_threshold", 0.5))

        probabilities = stage1.predict_proba(features)[:, 1]
        stage2_predictions = np.maximum(stage2.predict(features), 0.0)
        predicted_nonzero = probabilities >= threshold

        prediction = np.zeros(len(features), dtype=float)
        prediction[predicted_nonzero] = stage2_predictions[predicted_nonzero]

        return pd.DataFrame(
            {
                "predicted_total_work": prediction,
                "stage1_positive_probability": probabilities,
                "stage1_predicted_nonzero": predicted_nonzero.astype(int),
                "stage1_threshold": threshold,
            },
            index=features_df.index,
        )

    if "model" in model_bundle:
        model = model_bundle["model"]
        prediction = np.maximum(model.predict(features), 0.0)
        return pd.DataFrame({"predicted_total_work": prediction}, index=features_df.index)

    raise ValueError("Unsupported model artifact format. Expected two-stage or single-stage model bundle.")
