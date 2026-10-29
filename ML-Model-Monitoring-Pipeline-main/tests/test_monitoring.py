"""
tests/test_monitoring.py
Unit tests for the monitoring components.

Run:  pytest tests/ -v
"""
import json

import pytest
import pandas as pd

import config
from data.generate import _make_churn_dataset
from monitoring.performance_tracker import (
    compute_performance_metrics,
    check_performance_thresholds,
)
from monitoring.alerting import send_info_alert, load_recent_alerts


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def reference_df():
    return _make_churn_dataset(200, seed=42, drift=False)


@pytest.fixture
def drifted_df():
    return _make_churn_dataset(200, seed=99, drift=True, drift_strength=0.5)


@pytest.fixture
def nodrift_df():
    return _make_churn_dataset(200, seed=101, drift=False)


# ── Data generation tests ─────────────────────────────────────────────────────

def test_generate_reference(reference_df):
    assert len(reference_df) == 200
    assert set(config.FEATURE_COLUMNS).issubset(reference_df.columns)
    assert config.TARGET_COLUMN in reference_df.columns


def test_drifted_data_has_higher_charges(reference_df, drifted_df):
    """Drifted data should have higher monthly charges (by design)."""
    assert drifted_df["monthly_charges"].mean() > reference_df["monthly_charges"].mean()


def test_drifted_data_lower_tenure(reference_df, drifted_df):
    """Drifted data should have shorter tenure."""
    assert drifted_df["tenure"].mean() < reference_df["tenure"].mean()


def test_no_missing_values(reference_df):
    assert reference_df.isnull().sum().sum() == 0


# ── Performance metric tests ──────────────────────────────────────────────────

def test_compute_metrics_perfect():
    y = pd.Series([1, 0, 1, 0, 1])
    metrics = compute_performance_metrics(y, y)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_compute_metrics_partial():
    y_true = pd.Series([1, 1, 0, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 1, 1, 0])
    metrics = compute_performance_metrics(y_true, y_pred)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert "positive_rate" in metrics


def test_threshold_check_pass():
    metrics = {"accuracy": 0.90, "f1_score": 0.85}
    checks = check_performance_thresholds(metrics)
    assert all(v["passed"] for v in checks.values())


def test_threshold_check_fail():
    metrics = {"accuracy": 0.60, "f1_score": 0.55}
    checks = check_performance_thresholds(metrics)
    assert not checks["accuracy"]["passed"]
    assert not checks["f1_score"]["passed"]


# ── Alerting tests ────────────────────────────────────────────────────────────

def test_write_and_load_alert(tmp_path, monkeypatch):
    import monitoring.alerting as alerting_mod

    log_path = tmp_path / "alerts.jsonl"
    monkeypatch.setattr(alerting_mod, "ALERTS_LOG", log_path)

    send_info_alert("Test alert", batch_id="test123")
    alerts = load_recent_alerts(10)

    assert len(alerts) == 1
    assert alerts[0]["message"] == "Test alert"
    assert alerts[0]["level"] == "INFO"
    assert alerts[0]["batch_id"] == "test123"

    lines = log_path.read_text().splitlines()
    assert json.loads(lines[0])["message"] == "Test alert"


# ── Feature column consistency ────────────────────────────────────────────────

def test_feature_columns_present(reference_df):
    for col in config.FEATURE_COLUMNS:
        assert col in reference_df.columns, f"Missing feature: {col}"


def test_target_column_binary(reference_df):
    assert set(reference_df[config.TARGET_COLUMN].unique()).issubset({0, 1})
