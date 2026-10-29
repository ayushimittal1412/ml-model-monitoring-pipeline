"""
api/main.py
FastAPI serving layer for the churn prediction model.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.predict import load_production_model, predict_single


class ChurnFeatures(BaseModel):
    tenure: float = Field(..., ge=0, le=72)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_products: int = Field(..., ge=1, le=5)
    support_calls: int = Field(..., ge=0)
    late_payments: int = Field(..., ge=0)
    contract_type: int = Field(..., ge=0, le=2)
    internet_service: int = Field(..., ge=0, le=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_production_model()
    except Exception:
        pass
    yield


app = FastAPI(
    title="ML Monitor — Churn API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model": config.REGISTERED_MODEL_NAME}


@app.post("/predict")
def predict_endpoint(features: ChurnFeatures):
    try:
        return predict_single(features.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available. Train first: python models/train.py — {exc}",
        ) from exc
