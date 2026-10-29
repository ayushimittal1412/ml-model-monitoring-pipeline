"""
models/predict.py
Load the Production model from MLflow registry and run batch predictions.
Also used by the FastAPI serving layer.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from functools import lru_cache

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


@lru_cache(maxsize=1)
def load_production_model():
    """Load the latest Production model from MLflow registry (cached)."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{config.REGISTERED_MODEL_NAME}/Production"
    logger.info(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded successfully.")
    return model


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on a DataFrame with FEATURE_COLUMNS.
    Returns the input df with added columns: prediction, probability.
    """
    model = load_production_model()
    X = df[config.FEATURE_COLUMNS]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["prediction"] = preds
    result["probability"] = probs.round(4)
    return result


def predict_single(features: dict) -> dict:
    """Predict churn for a single customer (used by FastAPI)."""
    df = pd.DataFrame([features])
    result = predict(df)
    return {
        "prediction": int(result["prediction"].iloc[0]),
        "probability": float(result["probability"].iloc[0]),
        "churn_risk": (
            "HIGH" if result["probability"].iloc[0] > 0.7
            else "MEDIUM" if result["probability"].iloc[0] > 0.4
            else "LOW"
        ),
    }
