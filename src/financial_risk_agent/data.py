from __future__ import annotations

from pathlib import Path

import pandas as pd

from financial_risk_agent.config import FEATURE_COLUMNS, TARGET_COLUMN


class DatasetError(ValueError):
    """Raised when the input dataset does not match the expected UCI schema."""


def load_credit_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put UCI_Credit_Card.csv under data/raw/."
        )

    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise DatasetError(f"CSV is missing required columns: {missing}")

    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))

    return df


def normalize_application(payload: dict) -> dict[str, float]:
    """Return a complete feature dict, filling missing fields with safe defaults."""
    normalized: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        value = payload.get(col, 0)
        try:
            normalized[col] = float(value)
        except (TypeError, ValueError):
            normalized[col] = 0.0
    return normalized
