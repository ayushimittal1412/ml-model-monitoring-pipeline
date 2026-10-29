"""
scripts/setup.py
One-command project setup:
  1. Generate all datasets
  2. Train model + register to MLflow
  3. Run first monitoring pass
  4. Print next steps

Run:  python scripts/setup.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def main():
    logger.info("=" * 60)
    logger.info("ML Monitoring Pipeline — Full Setup")
    logger.info("=" * 60)

    # 1. Generate data
    logger.info("\n[1/3] Generating synthetic datasets…")
    from data.generate import generate_all
    paths = generate_all()
    for name, path in paths.items():
        logger.info(f"  ✓ {name}: {path}")

    # 2. Train + register model
    logger.info("\n[2/3] Training and registering model…")
    from models.train import train_and_register
    run_id = train_and_register()
    logger.info(f"  ✓ Best model registered (run_id={run_id})")

    # 3. First monitoring run
    logger.info("\n[3/3] Running initial monitoring pipeline (with drift)…")
    from pipelines.monitoring_pipeline import monitoring_pipeline
    summary = monitoring_pipeline(use_drifted_data=True)

    logger.info("\n" + "=" * 60)
    logger.info("Setup complete!")
    logger.info(f"  Drift detected : {summary['drift_detected']}")
    logger.info(f"  Drifted features: {summary['drifted_features']}")
    logger.info(f"  Alerts fired   : {summary['alerts_fired']}")
    logger.info("\n  ── Next steps ──")
    logger.info("  Dashboard   : streamlit run dashboard/app.py")
    logger.info("  MLflow UI   : mlflow ui --backend-store-uri sqlite:///mlflow.db")
    logger.info("  Prefect UI  : prefect server start  (then run deploy.py)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
