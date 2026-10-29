# ⬡ ML Model Monitoring Pipeline

End-to-end MLOps monitoring system: data drift detection, model performance tracking, automated alerting, and auto-retraining — orchestrated with Prefect, tracked in MLflow, and analysed with Evidently AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?style=flat-square)
![Prefect](https://img.shields.io/badge/Prefect-2.19-3B35E0?style=flat-square)
![Evidently](https://img.shields.io/badge/Evidently-0.4-FF5C5C?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## ✨ Features

| Feature | Details |
|---|---|
| 📊 Data drift detection | Evidently DataDriftPreset — per-feature PSI, KS test, Jensen-Shannon |
| 🎯 Prediction drift | Tracks whether model output distribution shifts over time |
| 📉 Performance tracking | Accuracy, F1, AUC logged per batch to MLflow |
| 🚨 Alerting system | Structured alerts (WARNING / CRITICAL) saved to JSONL log |
| ⚙️ Prefect orchestration | Tasks, flows, concurrent execution, scheduled deployments |
| 🔬 MLflow tracking | Every run, metric, and Evidently HTML report logged |
| 🔁 Auto-retraining | Prefect flow triggered when drift > threshold |
| 🖥 Streamlit dashboard | Live monitoring UI with trend charts and alert feed |
| 🧪 Test suite | pytest unit tests for all monitoring components |
| 🧬 Synthetic data | Realistic churn dataset with injected drift |

---

## 📁 Project Structure

```
ml-monitor/
├── config.py                       ← All settings + thresholds
├── .env.example                    → copy to .env
├── requirements.txt
│
├── data/
│   └── generate.py                 ← Synthetic churn data (with/without drift)
│
├── models/
│   ├── train.py                    ← Train 3 models, register best in MLflow
│   └── predict.py                  ← Load Production model, run inference
│
├── monitoring/
│   ├── drift_detector.py           ← Evidently reports + test suites
│   ├── performance_tracker.py      ← Metrics + threshold checks
│   └── alerting.py                 ← Structured alert generation + JSONL log
│
├── pipelines/
│   ├── monitoring_pipeline.py      ← Main Prefect flow (concurrent tasks)
│   └── retrain_pipeline.py         ← Auto-retrain Prefect flow
│
├── dashboard/
│   └── app.py                      ← Streamlit monitoring dashboard
│
├── api/
│   └── main.py                     ← FastAPI prediction endpoint
│
├── scripts/
│   ├── setup.py                    ← One-command full setup
│   └── deploy.py                   ← Prefect scheduled deployments
│
└── tests/
    └── test_monitoring.py          ← pytest unit tests
```

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/YOUR_USERNAME/ml-monitor.git
cd ml-monitor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. One-command setup (data + train + first monitoring run)

```bash
python scripts/setup.py
```

This will:
- Generate train / reference / production datasets
- Train 3 models (LogReg, RF, GBM), register best in MLflow
- Run the monitoring pipeline on drifted production data
- Print all results to console

### 3. View dashboards and API

```bash

streamlit run dashboard/app.py

uvicorn api.main:app --host 0.0.0.0 --port 8000

mlflow ui --backend-store-uri sqlite:///mlflow.db

prefect server start
```

Example API request after training:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure":24,"monthly_charges":75,"total_charges":1800,"num_products":2,"support_calls":1,"late_payments":0,"contract_type":0,"internet_service":2}'
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## 🏗 Architecture

```
Production data (batch)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Prefect Flow: monitoring_pipeline                    │
│                                                       │
│  ┌─────────────┐    ┌──────────────────────────────┐  │
│  │ Load data   │───▶│ Run predictions (MLflow model)│  │
│  └─────────────┘    └──────────────┬───────────────┘  │
│                                    │                  │
│              ┌─────────────────────┤                  │
│              ▼                     ▼                  │
│  ┌────────────────────┐ ┌────────────────────────┐   │
│  │ detect_data_drift  │ │ track_performance      │   │
│  │ (Evidently AI)     │ │ (sklearn metrics)      │   │
│  └────────┬───────────┘ └──────────┬─────────────┘   │
│           │  (concurrent)          │                  │
│           └─────────────┬──────────┘                  │
│                         ▼                             │
│              ┌────────────────────┐                   │
│              │ evaluate_and_alert │                   │
│              └────────┬───────────┘                   │
│                       │                               │
│              ┌────────▼───────────┐                   │
│              │ log_to_mlflow      │                   │
│              └────────────────────┘                   │
└───────────────────────────────────────────────────────┘
        │
        ▼
   alerts.jsonl  +  MLflow  +  Evidently HTML reports
        │
        ▼
  Streamlit Dashboard  /  Auto-retrain trigger
```

---

## 📊 Monitoring Signals

### Data Drift (Evidently)
- DataDriftPreset — tests each feature for statistical drift
- DataQualityPreset — checks nulls, out-of-range values, schema
- TargetDriftPreset — monitors prediction distribution shift
- Each report saved as HTML artifact in MLflow

### Performance Metrics (per batch)
| Metric | Threshold | Action |
|---|---|---|
| Accuracy | ≥ 0.75 | Alert if below |
| F1 Score | ≥ 0.70 | Alert if below |
| Prediction drift | ≤ 0.15 | Alert if above |

### Drift Share
| Threshold | Level |
|---|---|
| > 30% features drifted | WARNING |
| > 60% features drifted | CRITICAL |

---

## ⚙️ Scheduling with Prefect

```bash

prefect server start

python scripts/deploy.py


prefect worker start -p default-agent-pool
```

Then visit http://localhost:4200 to see the Prefect UI.

---

## 🛠 Possible Extensions

- [ ] Slack / email notifications on CRITICAL alerts
- [ ] Multi-model registry support
- [ ] Batch prediction logging to database
- [ ] Feature importance drift (SHAP)
- [ ] A/B model comparison in production
- [ ] Docker + Kubernetes deployment
- [ ] GitHub Actions CI/CD with model quality gates


