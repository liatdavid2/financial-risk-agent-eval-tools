from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# In Docker this is /app. In local editable install this is the repo root.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
GOLDEN_DATA_PATH = PROJECT_ROOT / "data" / "golden" / "golden_default_credit.jsonl"

RISK_MODEL_PATH = Path(os.getenv("RISK_MODEL_PATH", PROJECT_ROOT / "models" / "risk_model.joblib"))
SIMILAR_CASES_INDEX_PATH = Path(
    os.getenv("SIMILAR_CASES_INDEX_PATH", PROJECT_ROOT / "models" / "similar_cases_index.joblib")
)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

TARGET_COLUMN = "default.payment.next.month"

LANGGRAPH_TOOL_SEQUENCE = [
    "normalize_application",
    "credit_risk_model_predict",
    "payment_behavior_rules",
    "similar_cases_search",
    "policy_decision",
    "explanation_generator",
]
