"""
config.py
All tuneable settings live here. Override via environment variables or .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(exist_ok=True)

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "ml-monitoring-demo")
REGISTERED_MODEL_NAME: str = "churn-predictor"

# ── Prefect ───────────────────────────────────────────────────────────────────
PREFECT_API_URL: str = os.getenv("PREFECT_API_URL", "")  # empty = local

# ── Data drift thresholds ─────────────────────────────────────────────────────
# PSI (Population Stability Index): >0.1 = minor drift, >0.2 = major drift
PSI_THRESHOLD: float = float(os.getenv("PSI_THRESHOLD", "0.1"))
# KS test p-value: below this → statistically significant drift
KS_PVALUE_THRESHOLD: float = float(os.getenv("KS_PVALUE_THRESHOLD", "0.05"))
# Percentage of drifted features to trigger an alert
DRIFT_SHARE_THRESHOLD: float = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.3"))

# ── Model performance thresholds ──────────────────────────────────────────────
MIN_ACCURACY: float = float(os.getenv("MIN_ACCURACY", "0.75"))
MIN_F1: float = float(os.getenv("MIN_F1", "0.70"))
MAX_PREDICTION_DRIFT: float = float(os.getenv("MAX_PREDICTION_DRIFT", "0.15"))

# ── Monitoring schedule ───────────────────────────────────────────────────────
# How many new prediction rows trigger a monitoring run
MONITORING_BATCH_SIZE: int = int(os.getenv("MONITORING_BATCH_SIZE", "500"))
# Cron for scheduled Prefect runs (default: every hour)
MONITORING_CRON: str = os.getenv("MONITORING_CRON", "0 * * * *")

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# ── Dataset (demo synthetic data) ─────────────────────────────────────────────
N_TRAIN_SAMPLES: int = 5000
N_REFERENCE_SAMPLES: int = 1000
N_PRODUCTION_SAMPLES: int = 500
RANDOM_SEED: int = 42

# Feature names used throughout the project
FEATURE_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_products",
    "support_calls",
    "late_payments",
    "contract_type",   # ordinal encoded
    "internet_service", # ordinal encoded
]
TARGET_COLUMN = "churn"
