from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from financial_risk_agent.agent import cache_status, investigate_application


class ApplicationPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    customer_id: str | None = None


app = FastAPI(
    title="Financial Risk Agent Eval Tools",
    description="Credit risk investigation agent with LangGraph tools, Redis caching, and GenAI evaluation.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/cache/status")
def get_cache_status() -> dict[str, Any]:
    return cache_status()


@app.get("/sample-payload")
def sample_payload() -> dict[str, Any]:
    return {
        "customer_id": "C-DEMO-001",
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 3,
        "PAY_2": 2,
        "PAY_3": 2,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 18500,
        "BILL_AMT2": 17000,
        "BILL_AMT3": 16000,
        "BILL_AMT4": 15000,
        "BILL_AMT5": 14000,
        "BILL_AMT6": 13000,
        "PAY_AMT1": 400,
        "PAY_AMT2": 300,
        "PAY_AMT3": 500,
        "PAY_AMT4": 600,
        "PAY_AMT5": 700,
        "PAY_AMT6": 800,
    }


@app.post("/predict")
def predict(payload: ApplicationPayload) -> dict[str, Any]:
    try:
        return investigate_application(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
