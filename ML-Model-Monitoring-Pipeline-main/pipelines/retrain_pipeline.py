"""
pipelines/retrain_pipeline.py
Prefect flow that re-trains the model when monitoring detects significant drift.
Can be triggered programmatically from the monitoring pipeline or scheduled.

Run:  python pipelines/retrain_pipeline.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import mlflow
from loguru import logger
from prefect import flow, task, get_run_logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.train import train_and_register
from monitoring.alerting import send_info_alert


@task(name="check-retrain-needed")
def check_retrain_needed(drift_share: float, performance_f1: float) -> bool:
    """Decide whether retraining is warranted based on monitoring signals."""
    log = get_run_logger()
    retrain = False
    reasons = []

    if drift_share > config.DRIFT_SHARE_THRESHOLD:
        retrain = True
        reasons.append(f"drift_share={drift_share:.1%} > threshold={config.DRIFT_SHARE_THRESHOLD:.1%}")

    if performance_f1 < config.MIN_F1:
        retrain = True
        reasons.append(f"f1={performance_f1:.4f} < threshold={config.MIN_F1}")

    if retrain:
        log.warning(f"Retraining triggered: {'; '.join(reasons)}")
    else:
        log.info("No retraining needed.")

    return retrain


@task(name="retrain-model", retries=1, retry_delay_seconds=30)
def retrain_model() -> str:
    """Re-run the full training pipeline and register the new model."""
    log = get_run_logger()
    log.info("Starting model retraining…")
    run_id = train_and_register()
    log.info(f"Retraining complete. New MLflow run: {run_id}")
    return run_id


@task(name="validate-new-model")
def validate_new_model(run_id: str) -> bool:
    """Check that the newly trained model meets minimum quality gates."""
    log = get_run_logger()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    run = client.get_run(run_id)
    f1 = float(run.data.metrics.get("f1_score", 0))
    acc = float(run.data.metrics.get("accuracy", 0))

    passed = f1 >= config.MIN_F1 and acc >= config.MIN_ACCURACY
    log.info(f"Validation: f1={f1:.4f}, acc={acc:.4f}, passed={passed}")
    return passed


@flow(name="auto-retrain-pipeline")
def retrain_pipeline(
    drift_share: float = 0.0,
    performance_f1: float = 1.0,
    force: bool = False,
) -> dict:
    """
    Auto-retrain flow.

    Parameters
    ----------
    drift_share     : float  from monitoring pipeline
    performance_f1  : float  from monitoring pipeline
    force           : bool   skip the need check and retrain anyway
    """
    log = get_run_logger()
    batch_id = f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    should_retrain = force or check_retrain_needed(drift_share, performance_f1)

    if not should_retrain:
        send_info_alert("Auto-retrain skipped — no significant drift or degradation.", batch_id)
        return {"retrained": False, "reason": "thresholds_not_breached"}

    log.info("Proceeding with model retraining…")
    send_info_alert("Auto-retrain started.", batch_id, alert_type="retrain_start")

    new_run_id = retrain_model()
    is_valid = validate_new_model(new_run_id)

    if is_valid:
        send_info_alert(
            f"Model retrained and validated successfully (run={new_run_id}).",
            batch_id, alert_type="retrain_success",
        )
        return {"retrained": True, "run_id": new_run_id, "validated": True}
    else:
        send_info_alert(
            f"Retrained model FAILED validation — keeping previous version (run={new_run_id}).",
            batch_id, alert_type="retrain_failed",
        )
        return {"retrained": True, "run_id": new_run_id, "validated": False}


if __name__ == "__main__":
    result = retrain_pipeline(drift_share=0.5, performance_f1=0.65, force=False)
    print(result)
