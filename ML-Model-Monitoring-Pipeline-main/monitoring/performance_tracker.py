"""
monitoring/performance_tracker.py
Track and log model performance metrics over time to MLflow.
Compares current batch metrics to baseline from training.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def compute_performance_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: Optional[pd.Series] = None,
) -> dict:
    """Compute classification performance metrics for a batch."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = float("nan")

    # Churn rate in this batch vs expected
    metrics["positive_rate"] = float(y_pred.mean())
    metrics["actual_positive_rate"] = float(y_true.mean())

    return {k: round(float(v), 4) for k, v in metrics.items()}


def check_performance_thresholds(metrics: dict) -> dict:
    """
    Compare metrics against configured thresholds.
    Returns a dict of {metric: {"value", "threshold", "passed"}}.
    """
    checks = {}
    threshold_map = {
        "accuracy": (config.MIN_ACCURACY, "gte"),
        "f1_score": (config.MIN_F1, "gte"),
    }

    for metric, (threshold, direction) in threshold_map.items():
        value = metrics.get(metric, 0.0)
        passed = value >= threshold if direction == "gte" else value <= threshold
        checks[metric] = {
            "value": value,
            "threshold": threshold,
            "passed": passed,
        }
        if not passed:
            logger.warning(
                f"⚠ Performance alert: {metric}={value:.4f} "
                f"(threshold={'≥' if direction=='gte' else '≤'}{threshold})"
            )

    return checks


def log_performance_to_mlflow(
    metrics: dict,
    threshold_checks: dict,
    batch_id: str,
    run_name: Optional[str] = None,
) -> str:
    """Log performance metrics and alerts to MLflow. Returns run_id."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    alerts = [k for k, v in threshold_checks.items() if not v["passed"]]

    with mlflow.start_run(run_name=run_name or f"perf_monitor_{batch_id}", nested=True) as run:
        mlflow.set_tag("monitoring_type", "performance")
        mlflow.set_tag("batch_id", batch_id)
        mlflow.set_tag("has_alerts", str(len(alerts) > 0))
        mlflow.set_tag("alert_metrics", ",".join(alerts) if alerts else "none")

        mlflow.log_metrics(metrics)

        # Log threshold results
        for metric, check in threshold_checks.items():
            mlflow.log_metric(f"{metric}_passed", int(check["passed"]))

        logger.info(
            f"Performance logged (run={run.info.run_id}): "
            f"acc={metrics.get('accuracy', 0):.4f}  "
            f"f1={metrics.get('f1_score', 0):.4f}"
        )
        return run.info.run_id
