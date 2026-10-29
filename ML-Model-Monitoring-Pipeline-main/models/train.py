"""
models/train.py
Train a churn prediction model, log everything to MLflow, and register
the best model in the MLflow Model Registry.

Run:  python models/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.generate import generate_all

MODELS = {
    "logistic_regression": LogisticRegression(
        C=1.0, max_iter=500, random_state=config.RANDOM_SEED
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=config.RANDOM_SEED
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        random_state=config.RANDOM_SEED,
    ),
}


def _load_or_generate_train_data() -> pd.DataFrame:
    train_path = config.DATA_DIR / "train.parquet"
    if not train_path.exists():
        logger.info("Training data not found — generating…")
        generate_all()
    return pd.read_parquet(train_path)


def train_and_register():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    df = _load_or_generate_train_data()
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
    )

    best_run_id = None
    best_f1 = 0.0

    for model_name, estimator in MODELS.items():
        logger.info(f"Training: {model_name}")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])

        with mlflow.start_run(run_name=model_name) as run:
            # Cross-validation
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)

            # Final fit
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            cv_mean = cv_scores.mean()

            # Log params
            mlflow.log_params({
                "model_type": model_name,
                "n_features": len(config.FEATURE_COLUMNS),
                "train_size": len(X_train),
                "test_size": len(X_test),
                **estimator.get_params(),
            })

            # Log metrics
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_score": f1,
                "roc_auc": auc,
                "cv_f1_mean": cv_mean,
                "cv_f1_std": cv_scores.std(),
            })

            # Log classification report
            report = classification_report(y_test, y_pred)
            mlflow.log_text(report, "classification_report.txt")

            # Log model
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                registered_model_name=None,  # register best only
            )

            logger.info(
                f"  acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  cv_f1={cv_mean:.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_run_id = run.info.run_id

    # ── Register best model ───────────────────────────────────────────────────
    if best_run_id:
        logger.info(f"Registering best model (run={best_run_id}, f1={best_f1:.4f})")
        model_uri = f"runs:/{best_run_id}/model"
        mv = mlflow.register_model(
            model_uri=model_uri,
            name=config.REGISTERED_MODEL_NAME,
        )
        logger.info(
            f"Registered '{config.REGISTERED_MODEL_NAME}' version {mv.version}"
        )

        # Transition to Production stage
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=config.REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"Model transitioned to 'Production'")

    return best_run_id


if __name__ == "__main__":
    train_and_register()
