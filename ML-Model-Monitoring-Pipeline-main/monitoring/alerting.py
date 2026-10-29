"""
monitoring/alerting.py
Generates structured alerts when drift or performance thresholds are breached.
Alerts are logged to MLflow tags, a local JSON log, and printed to stdout.

Extend this module to send to Slack, PagerDuty, email, etc.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

ALERTS_LOG = config.BASE_DIR / "alerts.jsonl"

AlertLevel = Literal["INFO", "WARNING", "CRITICAL"]


def _write_alert(alert: dict):
    with open(ALERTS_LOG, "a") as f:
        f.write(json.dumps(alert) + "\n")


def _format_alert(alert: dict) -> str:
    icon = {"INFO": "ℹ", "WARNING": "⚠", "CRITICAL": "🚨"}.get(alert["level"], "•")
    return (
        f"{icon} [{alert['level']}] {alert['timestamp'][:19]}  "
        f"| {alert['type'].upper()} | {alert['message']}"
    )


def send_drift_alert(
    drift_summary: dict,
    batch_id: str,
    level: AlertLevel = "WARNING",
):
    """Generate and log a data drift alert."""
    drifted = drift_summary.get("drifted_features", [])
    share = drift_summary.get("drift_share", 0)

    if level == "CRITICAL" or share > config.PSI_THRESHOLD * 2:
        level = "CRITICAL"

    message = (
        f"Data drift detected in {len(drifted)} feature(s) "
        f"({share:.0%} of features). "
        f"Drifted: {', '.join(drifted[:5])}{'...' if len(drifted) > 5 else ''}"
    )

    alert = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "data_drift",
        "level": level,
        "batch_id": batch_id,
        "message": message,
        "details": {
            "drift_share": share,
            "drifted_features": drifted,
            "feature_scores": drift_summary.get("feature_drift_scores", {}),
        },
    }

    _write_alert(alert)
    logger.warning(_format_alert(alert))
    return alert


def send_performance_alert(
    threshold_checks: dict,
    batch_id: str,
    level: AlertLevel = "WARNING",
):
    """Generate and log a model performance degradation alert."""
    failed = [k for k, v in threshold_checks.items() if not v["passed"]]

    details_str = ", ".join(
        f"{m}={threshold_checks[m]['value']:.3f} (≥{threshold_checks[m]['threshold']})"
        for m in failed
    )
    message = f"Performance degradation: {details_str}"

    alert = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "performance_degradation",
        "level": level,
        "batch_id": batch_id,
        "message": message,
        "details": {m: threshold_checks[m] for m in failed},
    }

    _write_alert(alert)
    logger.warning(_format_alert(alert))
    return alert


def send_info_alert(message: str, batch_id: str, alert_type: str = "info"):
    """Log a general informational alert."""
    alert = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": alert_type,
        "level": "INFO",
        "batch_id": batch_id,
        "message": message,
        "details": {},
    }
    _write_alert(alert)
    logger.info(_format_alert(alert))
    return alert


def load_recent_alerts(n: int = 50) -> list[dict]:
    """Load the most recent n alerts from the JSONL log."""
    if not ALERTS_LOG.exists():
        return []
    lines = ALERTS_LOG.read_text().strip().splitlines()
    alerts = [json.loads(l) for l in lines if l.strip()]
    return alerts[-n:]
