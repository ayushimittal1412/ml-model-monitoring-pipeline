"""
pipelines/monitoring_pipeline.py
===================================
Prefect 2.x monitoring pipeline.

Flow: load_data → run_predictions → detect_drift → track_performance → alert

Run once:    python pipelines/monitoring_pipeline.py
Schedule:    prefect deployment build + apply (see scripts/deploy.py)
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from loguru import logger
from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.generate import generate_all
from models.predict import predict
from monitoring.drift_detector import (
    run_data_drift_report,
    run_drift_test_suite,
    run_prediction_drift_report,
)
from monitoring.performance_tracker import (
    check_performance_thresholds,
    compute_performance_metrics,
    log_performance_to_mlflow,
)
from monitoring.alerting import (
    send_drift_alert,
    send_performance_alert,
    send_info_alert,
)
from pipelines.retrain_pipeline import retrain_pipeline


# ══════════════════════════════════════════════════════════════════════════════
# TASKS
# ══════════════════════════════════════════════════════════════════════════════

@task(name="load-reference-data", retries=2, retry_delay_seconds=5)
def load_reference_data() -> pd.DataFrame:
    """Load the reference (training distribution) dataset."""
    log = get_run_logger()
    ref_path = config.DATA_DIR / "reference.parquet"
    if not ref_path.exists():
        log.warning("Reference data not found — generating all datasets first.")
        generate_all()
    df = pd.read_parquet(ref_path)
    log.info(f"Reference data: {len(df):,} rows, {df.shape[1]} cols")
    return df


@task(name="load-production-data", retries=2, retry_delay_seconds=5)
def load_production_data(use_drift: bool = True) -> pd.DataFrame:
    """Load the current production batch."""
    log = get_run_logger()
    fname = "production_drift.parquet" if use_drift else "production_nodrift.parquet"
    prod_path = config.DATA_DIR / fname
    if not prod_path.exists():
        generate_all()
    df = pd.read_parquet(prod_path)
    log.info(f"Production data ({fname}): {len(df):,} rows")
    return df


@task(name="run-predictions", retries=1)
def run_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Run the production model on the current batch."""
    log = get_run_logger()
    result = predict(df)
    pos_rate = result["prediction"].mean()
    log.info(
        f"Predictions complete: {len(result):,} rows, "
        f"predicted churn rate={pos_rate:.1%}"
    )
    return result


@task(name="detect-data-drift")
def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    mlflow_run_id: str,
) -> dict:
    """Run Evidently data drift detection."""
    log = get_run_logger()

    drift_summary = run_data_drift_report(
        reference=reference,
        current=current,
        run_id=mlflow_run_id,
        save_html=True,
    )

    test_results = run_drift_test_suite(
        reference=reference,
        current=current,
        run_id=mlflow_run_id,
    )

    log.info(
        f"Drift: detected={drift_summary['dataset_drift_detected']}  "
        f"share={drift_summary['drift_share']:.1%}  "
        f"tests: {test_results['passed']}/{test_results['total_tests']} passed"
    )
    return {**drift_summary, "test_results": test_results}


@task(name="detect-prediction-drift")
def detect_prediction_drift(
    reference_with_preds: pd.DataFrame,
    current_with_preds: pd.DataFrame,
    mlflow_run_id: str,
) -> dict:
    """Check if the model's prediction distribution has shifted."""
    return run_prediction_drift_report(
        reference=reference_with_preds,
        current=current_with_preds,
        run_id=mlflow_run_id,
        save_html=True,
    )


@task(name="track-model-performance")
def track_model_performance(
    current_with_preds: pd.DataFrame,
    batch_id: str,
    mlflow_run_id: str,
) -> dict:
    """
    Compute performance metrics if ground truth (churn) is available.
    In production, labels may arrive with a delay — this simulates
    the case where we have them for the current batch.
    """
    log = get_run_logger()

    if config.TARGET_COLUMN not in current_with_preds.columns:
        log.warning("No ground-truth labels in current batch — skipping performance tracking.")
        return {}

    metrics = compute_performance_metrics(
        y_true=current_with_preds[config.TARGET_COLUMN],
        y_pred=current_with_preds["prediction"],
        y_prob=current_with_preds.get("probability"),
    )

    threshold_checks = check_performance_thresholds(metrics)

    log_performance_to_mlflow(
        metrics=metrics,
        threshold_checks=threshold_checks,
        batch_id=batch_id,
        run_name=f"perf_monitor_{batch_id[:8]}",
    )

    log.info(
        f"Performance: acc={metrics.get('accuracy', 0):.4f}  "
        f"f1={metrics.get('f1_score', 0):.4f}  "
        f"auc={metrics.get('roc_auc', 0):.4f}"
    )
    return {"metrics": metrics, "threshold_checks": threshold_checks}


@task(name="evaluate-and-alert")
def evaluate_and_alert(
    drift_results: dict,
    pred_drift_results: dict,
    performance_results: dict,
    batch_id: str,
) -> dict:
    """
    Evaluate all monitoring signals and fire alerts if thresholds are breached.
    Returns a summary dict.
    """
    log = get_run_logger()
    alerts_fired = []

    # ── Data drift alert ──────────────────────────────────────────────────────
    if drift_results.get("dataset_drift_detected"):
        share = drift_results.get("drift_share", 0)
        level = "CRITICAL" if share > config.DRIFT_SHARE_THRESHOLD else "WARNING"
        alert = send_drift_alert(drift_results, batch_id, level=level)
        alerts_fired.append(alert)
        log.warning(f"DRIFT ALERT fired: {len(drift_results.get('drifted_features', []))} features")

    # ── Prediction drift alert ────────────────────────────────────────────────
    if pred_drift_results.get("prediction_drift_detected"):
        alert = send_drift_alert(
            {"drifted_features": ["prediction"], "drift_share": 0.5,
             "feature_drift_scores": {"prediction": pred_drift_results.get("prediction_drift_score", 0)}},
            batch_id,
            level="WARNING",
        )
        alerts_fired.append(alert)

    # ── Performance alert ─────────────────────────────────────────────────────
    if performance_results:
        failed_checks = {
            k: v for k, v in performance_results.get("threshold_checks", {}).items()
            if not v["passed"]
        }
        if failed_checks:
            alert = send_performance_alert(failed_checks, batch_id, level="WARNING")
            alerts_fired.append(alert)

    # ── All clear ─────────────────────────────────────────────────────────────
    if not alerts_fired:
        send_info_alert(
            "All monitoring checks passed. No drift or degradation detected.",
            batch_id=batch_id,
            alert_type="all_clear",
        )

    summary = {
        "batch_id": batch_id,
        "timestamp": datetime.utcnow().isoformat(),
        "drift_detected": drift_results.get("dataset_drift_detected", False),
        "drift_share": drift_results.get("drift_share", 0),
        "drifted_features": drift_results.get("drifted_features", []),
        "prediction_drift": pred_drift_results.get("prediction_drift_detected", False),
        "performance": performance_results.get("metrics", {}),
        "alerts_fired": len(alerts_fired),
        "alert_levels": [a["level"] for a in alerts_fired],
    }

    log.info(
        f"Monitoring complete — "
        f"drift={summary['drift_detected']}  "
        f"alerts={summary['alerts_fired']}"
    )
    return summary


@task(name="log-monitoring-run-to-mlflow")
def log_monitoring_run(summary: dict, mlflow_run_id: str):
    """Log the complete monitoring summary as a single MLflow run."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    client.set_tag(mlflow_run_id, "monitoring_type", "full_pipeline")
    client.set_tag(mlflow_run_id, "batch_id", summary["batch_id"])
    client.set_tag(mlflow_run_id, "drift_detected", str(summary["drift_detected"]))
    client.set_tag(mlflow_run_id, "alerts_fired", str(summary["alerts_fired"]))

    client.log_metric(mlflow_run_id, "drift_share", summary.get("drift_share", 0))
    client.log_metric(mlflow_run_id, "n_drifted_features", len(summary.get("drifted_features", [])))
    client.log_metric(mlflow_run_id, "prediction_drift", int(summary.get("prediction_drift", False)))
    client.log_metric(mlflow_run_id, "alerts_count", summary.get("alerts_fired", 0))

    perf = summary.get("performance", {})
    for k, v in perf.items():
            try:
                client.log_metric(mlflow_run_id, f"perf_{k}", v)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# FLOW
# ══════════════════════════════════════════════════════════════════════════════

@flow(
    name="ml-monitoring-pipeline",
    description="End-to-end ML model monitoring: drift detection + performance tracking.",
    task_runner=ConcurrentTaskRunner(),
)
def monitoring_pipeline(use_drifted_data: bool = True, auto_retrain: bool = False) -> dict:
    """
    Main monitoring flow.

    Parameters
    ----------
    use_drifted_data : bool
        True  → load production_drift.parquet (simulates real drift)
        False → load production_nodrift.parquet (baseline check)
    auto_retrain : bool
        When True, run the retrain pipeline if drift or F1 crosses thresholds
    """
    batch_id = str(uuid.uuid4())[:12]
    log = get_run_logger()
    log.info(f"Starting monitoring run | batch_id={batch_id} | drifted={use_drifted_data}")

    # ── MLflow parent run ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"monitoring_{batch_id[:8]}") as mlflow_run:
        mlflow_run_id = mlflow_run.info.run_id

        # 1. Load data
        reference = load_reference_data()
        production = load_production_data(use_drift=use_drifted_data)

        # 2. Run model predictions on current batch
        production_with_preds = run_predictions(production)

        # Also add predictions to reference for prediction drift comparison
        reference_with_preds = run_predictions(reference)

        # 3. Detect drift (can run concurrently)
        drift_future = detect_data_drift.submit(reference, production, mlflow_run_id)
        pred_drift_future = detect_prediction_drift.submit(
            reference_with_preds, production_with_preds, mlflow_run_id
        )

        # 4. Track performance
        perf_future = track_model_performance.submit(
            production_with_preds, batch_id, mlflow_run_id
        )

        # 5. Collect results
        drift_results = drift_future.result()
        pred_drift_results = pred_drift_future.result()
        performance_results = perf_future.result()

        # 6. Evaluate and alert
        summary = evaluate_and_alert(
            drift_results, pred_drift_results, performance_results, batch_id
        )

        # 7. Log summary
        log_monitoring_run(summary, mlflow_run_id)

        # 8. Auto-retrain when drift or performance crosses thresholds
        drift_share = summary.get("drift_share", 0)
        perf_f1 = summary.get("performance", {}).get("f1_score", 1.0)
        if auto_retrain and (
            drift_share > config.DRIFT_SHARE_THRESHOLD or perf_f1 < config.MIN_F1
        ):
            log.warning(
                f"Triggering auto-retrain (drift_share={drift_share:.1%}, f1={perf_f1:.4f})"
            )
            summary["retrain"] = retrain_pipeline(
                drift_share=drift_share,
                performance_f1=perf_f1,
            )
        else:
            summary["retrain"] = {
                "retrained": False,
                "reason": "disabled" if not auto_retrain else "thresholds_not_breached",
            }

    log.info(f"Pipeline complete | run_id={mlflow_run_id}")
    return summary


if __name__ == "__main__":
    result = monitoring_pipeline(use_drifted_data=True, auto_retrain=True)
    print("\n── Monitoring Summary ──")
    import json
    print(json.dumps(result, indent=2, default=str))
