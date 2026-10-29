"""
scripts/deploy.py
Build and apply Prefect deployments for scheduled monitoring.

Run:  python scripts/deploy.py

Then start Prefect agent:  prefect agent start -q default
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

import config
from pipelines.monitoring_pipeline import monitoring_pipeline
from pipelines.retrain_pipeline import retrain_pipeline


def deploy_monitoring():
    deployment = Deployment.build_from_flow(
        flow=monitoring_pipeline,
        name="hourly-monitoring",
        schedule=CronSchedule(cron=config.MONITORING_CRON, timezone="UTC"),
        parameters={"use_drifted_data": True},
        tags=["monitoring", "production"],
        description="Scheduled ML monitoring: drift detection + performance tracking.",
    )
    deployment.apply()
    print(f"✓ Deployed 'hourly-monitoring' with cron: {config.MONITORING_CRON}")


def deploy_retrain():
    deployment = Deployment.build_from_flow(
        flow=retrain_pipeline,
        name="manual-retrain",
        parameters={"force": True},
        tags=["training", "retrain"],
        description="Manual or triggered model retraining pipeline.",
    )
    deployment.apply()
    print("✓ Deployed 'manual-retrain' (trigger manually from Prefect UI)")


if __name__ == "__main__":
    print("Building Prefect deployments…")
    deploy_monitoring()
    deploy_retrain()
    print("\nDone. Start your agent with:\n  prefect agent start -q default")
