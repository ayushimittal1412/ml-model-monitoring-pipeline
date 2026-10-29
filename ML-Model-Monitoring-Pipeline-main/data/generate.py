"""
data/generate.py
Generates synthetic customer-churn data:
  - Reference dataset  (training distribution)
  - Production dataset (with injected drift for realism)

Run standalone:  python data/generate.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _make_churn_dataset(
    n: int,
    seed: int,
    drift: bool = False,
    drift_strength: float = 0.3,
) -> pd.DataFrame:
    """
    Synthetic telecom churn dataset.
    `drift=True` shifts several feature distributions to simulate production drift.
    """
    rng = np.random.default_rng(seed)

    # ── Base features ─────────────────────────────────────────────────────────
    tenure = rng.normal(loc=32, scale=20, size=n).clip(0, 72)
    monthly_charges = rng.normal(loc=65, scale=25, size=n).clip(18, 120)
    total_charges = tenure * monthly_charges + rng.normal(0, 50, n)
    total_charges = total_charges.clip(0)
    num_products = rng.integers(1, 6, size=n)
    support_calls = rng.poisson(lam=1.5, size=n).clip(0, 10)
    late_payments = rng.integers(0, 8, size=n)
    contract_type = rng.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2])  # M2M, 1yr, 2yr
    internet_service = rng.choice([0, 1, 2], size=n, p=[0.1, 0.5, 0.4])  # None, DSL, Fiber

    if drift:
        # Simulate real drift: newer customers, higher charges, more support calls
        tenure *= (1 - drift_strength * 0.4)
        monthly_charges *= (1 + drift_strength * 0.25)
        support_calls = (support_calls * (1 + drift_strength * 0.5)).clip(0, 10).astype(int)
        # Shift contract distribution toward month-to-month (higher churn risk)
        contract_type = rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1])

    # ── Label: logistic churn probability ────────────────────────────────────
    log_odds = (
        -2.0
        + 0.03 * (60 - tenure)
        + 0.02 * (monthly_charges - 50)
        + 0.4 * support_calls
        + 0.3 * late_payments
        - 0.5 * contract_type
        + 0.1 * internet_service
        + rng.normal(0, 0.5, n)
    )
    prob_churn = 1 / (1 + np.exp(-log_odds))
    churn = (rng.uniform(size=n) < prob_churn).astype(int)

    df = pd.DataFrame({
        "tenure": tenure.round(1),
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "num_products": num_products,
        "support_calls": support_calls,
        "late_payments": late_payments,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "churn": churn,
    })
    return df


def generate_all(output_dir: Path | None = None) -> dict[str, Path]:
    """Generate and save all datasets. Returns dict of name → path."""
    out = output_dir or config.DATA_DIR

    # Reference  (clean training distribution)
    ref = _make_churn_dataset(
        config.N_REFERENCE_SAMPLES,
        seed=config.RANDOM_SEED,
        drift=False,
    )
    # Training   (larger, same distribution)
    train = _make_churn_dataset(
        config.N_TRAIN_SAMPLES,
        seed=config.RANDOM_SEED + 1,
        drift=False,
    )
    # Production (drifted — what the model sees in prod)
    prod = _make_churn_dataset(
        config.N_PRODUCTION_SAMPLES,
        seed=config.RANDOM_SEED + 99,
        drift=True,
        drift_strength=0.35,
    )
    # Production (no drift — for comparison tests)
    prod_nodrift = _make_churn_dataset(
        config.N_PRODUCTION_SAMPLES,
        seed=config.RANDOM_SEED + 200,
        drift=False,
    )

    paths = {}
    for name, df in [
        ("train", train),
        ("reference", ref),
        ("production_drift", prod),
        ("production_nodrift", prod_nodrift),
    ]:
        p = out / f"{name}.parquet"
        df.to_parquet(p, index=False)
        logger.info(f"Saved {name}: {len(df):,} rows → {p}")
        paths[name] = p

    return paths


if __name__ == "__main__":
    generate_all()
    logger.info("All datasets generated.")
