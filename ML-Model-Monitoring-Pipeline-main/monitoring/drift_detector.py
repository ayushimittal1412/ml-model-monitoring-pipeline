"""
monitoring/drift_detector.py
Data drift detection using Evidently AI.

Generates:
  - DataDriftPreset report  (per-feature drift tests)
  - DataQualityPreset report (nulls, outliers, schema)
  - TargetDriftPreset report (prediction/label drift)

Each report is saved as HTML + logged to MLflow as an artifact.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.test_suite import TestSuite
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _column_mapping() -> ColumnMapping:
    return ColumnMapping(
        target=config.TARGET_COLUMN,
        prediction="prediction",
        numerical_features=[
            "tenure", "monthly_charges", "total_charges",
            "num_products", "support_calls", "late_payments",
        ],
        categorical_features=["contract_type", "internet_service"],
    )


# ── Data Drift Report ─────────────────────────────────────────────────────────

def run_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    run_id: Optional[str] = None,
    save_html: bool = True,
) -> dict:
    """
    Generate a full Evidently data drift report.
    Returns a dict of summary metrics.
    Optionally logs the HTML artifact to MLflow.
    """
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetMissingValuesMetric(),
    ])

    col_map = _column_mapping()
    report.run(
        reference_data=reference[config.FEATURE_COLUMNS],
        current_data=current[config.FEATURE_COLUMNS],
        column_mapping=ColumnMapping(
            numerical_features=[
                "tenure", "monthly_charges", "total_charges",
                "num_products", "support_calls", "late_payments",
            ],
            categorical_features=["contract_type", "internet_service"],
        ),
    )

    # Extract summary from report JSON
    report_dict = report.as_dict()
    drift_result = _parse_drift_summary(report_dict)

    # Save HTML report
    if save_html:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = config.REPORTS_DIR / f"data_drift_{ts}.html"
        report.save_html(str(html_path))
        logger.info(f"Data drift report saved: {html_path}")

        if run_id:
            mlflow.MlflowClient().log_artifact(
                run_id, str(html_path), artifact_path="drift_reports"
            )

    return drift_result


# ── Prediction Drift Report ───────────────────────────────────────────────────

def run_prediction_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    run_id: Optional[str] = None,
    save_html: bool = True,
) -> dict:
    """Check if the model's prediction distribution has shifted."""
    # Both dataframes must have a 'prediction' column
    required_cols = config.FEATURE_COLUMNS + ["prediction"]
    ref_sub = reference[required_cols] if all(c in reference for c in required_cols) else reference
    cur_sub = current[required_cols] if all(c in current for c in required_cols) else current

    report = Report(metrics=[
        TargetDriftPreset(),
        ColumnDriftMetric(column_name="prediction"),
    ])

    report.run(
        reference_data=ref_sub,
        current_data=cur_sub,
        column_mapping=ColumnMapping(prediction="prediction"),
    )

    report_dict = report.as_dict()

    result = {
        "prediction_drift_detected": False,
        "prediction_drift_score": 0.0,
    }

    # Parse prediction drift score
    try:
        for metric in report_dict.get("metrics", []):
            if metric.get("metric") == "ColumnDriftMetric":
                result_data = metric.get("result", {})
                result["prediction_drift_detected"] = result_data.get("drift_detected", False)
                result["prediction_drift_score"] = result_data.get("stattest_threshold", 0.0)
                break
    except Exception as e:
        logger.warning(f"Could not parse prediction drift: {e}")

    if save_html:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = config.REPORTS_DIR / f"prediction_drift_{ts}.html"
        report.save_html(str(html_path))
        if run_id:
            mlflow.MlflowClient().log_artifact(
                run_id, str(html_path), artifact_path="drift_reports"
            )

    return result


# ── Test Suite ────────────────────────────────────────────────────────────────

def run_drift_test_suite(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    run_id: Optional[str] = None,
) -> dict:
    """
    Run Evidently test suite — pass/fail checks for data quality and drift.
    Returns summary of test results.
    """
    suite = TestSuite(tests=[
        DataDriftTestPreset(),
        DataQualityTestPreset(),
    ])

    suite.run(
        reference_data=reference[config.FEATURE_COLUMNS],
        current_data=current[config.FEATURE_COLUMNS],
        column_mapping=ColumnMapping(
            numerical_features=[
                "tenure", "monthly_charges", "total_charges",
                "num_products", "support_calls", "late_payments",
            ],
            categorical_features=["contract_type", "internet_service"],
        ),
    )

    suite_dict = suite.as_dict()
    total = len(suite_dict.get("tests", []))
    passed = sum(1 for t in suite_dict.get("tests", []) if t.get("status") == "SUCCESS")
    failed = total - passed

    # Save HTML
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = config.REPORTS_DIR / f"test_suite_{ts}.html"
    suite.save_html(str(html_path))

    if run_id:
        mlflow.MlflowClient().log_artifact(
            run_id, str(html_path), artifact_path="test_suites"
        )

    return {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "html_path": str(html_path),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_drift_summary(report_dict: dict) -> dict:
    """Extract key drift metrics from the Evidently report JSON."""
    result = {
        "dataset_drift_detected": False,
        "drift_share": 0.0,
        "drifted_features": [],
        "feature_drift_scores": {},
        "missing_values_share": 0.0,
    }

    try:
        for metric in report_dict.get("metrics", []):
            m = metric.get("metric", "")
            r = metric.get("result", {})

            if m == "DatasetDriftMetric":
                result["dataset_drift_detected"] = r.get("dataset_drift", False)
                result["drift_share"] = r.get("share_of_drifted_columns", 0.0)

            elif m == "DataDriftTable":
                for col, info in r.get("drift_by_columns", {}).items():
                    score = info.get("drift_score", 0.0)
                    drifted = info.get("drift_detected", False)
                    result["feature_drift_scores"][col] = round(score, 4)
                    if drifted:
                        result["drifted_features"].append(col)

            elif m == "DatasetMissingValuesMetric":
                result["missing_values_share"] = r.get(
                    "current", {}
                ).get("share_of_missing_values", 0.0)

    except Exception as e:
        logger.warning(f"Drift summary parse error: {e}")

    return result
