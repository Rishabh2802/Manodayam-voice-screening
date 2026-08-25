import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from feature_extraction import extract_audio_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

DEPRESSION_MODEL_PATH = MODEL_DIR / "depression_q1_logistic.joblib"
ANXIETY_MODEL_PATH = MODEL_DIR / "anxiety_q1_logistic.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns_q1.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


depression_model = joblib.load(DEPRESSION_MODEL_PATH)
anxiety_model = joblib.load(ANXIETY_MODEL_PATH)

with open(FEATURE_COLUMNS_PATH, "r") as f:
    FEATURE_COLUMNS_Q1 = json.load(f)

with open(METADATA_PATH, "r") as f:
    MODEL_METADATA = json.load(f)


def prepare_q1_feature_vector(audio_path):
    raw_features = extract_audio_features(audio_path)

    # Training columns were saved with q1_ prefix
    prefixed_features = {
        f"q1_{key}": value
        for key, value in raw_features.items()
    }

    row = pd.DataFrame([prefixed_features])

    for col in FEATURE_COLUMNS_Q1:
        if col not in row.columns:
            row[col] = np.nan

    row = row[FEATURE_COLUMNS_Q1]

    return row


def predict_single_model(model, X):
    prediction = int(model.predict(X)[0])

    probability = None

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]

        if len(prob) > 1:
            probability = float(prob[1])
        else:
            probability = float(prob[0])

    return prediction, probability


def label_result(prediction):
    if prediction == 1:
        return "Elevated symptoms"
    return "Minimal / No symptoms"


def probability_to_risk(probability):
    if probability is None:
        return "Not available"

    if probability >= 0.75:
        return "High"
    elif probability >= 0.50:
        return "Moderate"
    else:
        return "Low"


def predict_all(audio_path):
    X = prepare_q1_feature_vector(audio_path)

    depression_pred, depression_prob = predict_single_model(
        depression_model,
        X
    )

    anxiety_pred, anxiety_prob = predict_single_model(
        anxiety_model,
        X
    )

    result = {
        "input": "Q1 Daily Routine audio",
        "depression": {
            "prediction": depression_pred,
            "label": label_result(depression_pred),
            "probability_elevated": depression_prob,
            "risk_level": probability_to_risk(depression_prob),
        },
        "anxiety": {
            "prediction": anxiety_pred,
            "label": label_result(anxiety_pred),
            "probability_elevated": anxiety_prob,
            "risk_level": probability_to_risk(anxiety_prob),
        },
        "disclaimer": (
            "This is a speech-based screening output, not a clinical diagnosis. "
            "Please consult a qualified mental health professional for clinical evaluation."
        )
    }

    return result