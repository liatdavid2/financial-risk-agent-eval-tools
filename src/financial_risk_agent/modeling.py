from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from financial_risk_agent.config import FEATURE_COLUMNS, TARGET_COLUMN
from financial_risk_agent.data import load_credit_dataset


def train_credit_risk_model(
    csv_path: str | Path,
    model_path: str | Path,
    index_path: str | Path,
    reports_path: str | Path,
) -> dict[str, Any]:
    df = load_credit_dataset(csv_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=220,
                    max_depth=10,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "test_rows": int(len(X_test)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision_pr_auc": float(average_precision_score(y_test, proba)),
        "classification_report": classification_report(y_test, pred, output_dict=True),
        "feature_columns": FEATURE_COLUMNS,
    }

    model_path = Path(model_path)
    index_path = Path(index_path)
    reports_path = Path(reports_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    reports_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    # Similar-case index over the full dataset. This is a light-weight vector search demo
    # using standardized tabular vectors + cosine distance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLUMNS])
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(X_scaled)
    index_artifact = {
        "scaler": scaler,
        "nearest_neighbors": nn,
        "features": df[FEATURE_COLUMNS].reset_index(drop=True),
        "ids": df["ID"].astype(str).reset_index(drop=True),
        "labels": df[TARGET_COLUMN].astype(int).reset_index(drop=True),
    }
    joblib.dump(index_artifact, index_path)

    reports_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run: python scripts/train_model.py --csv data/raw/UCI_Credit_Card.csv"
        )
    return joblib.load(path)


def predict_default_probability(model, application: dict) -> float:
    row = pd.DataFrame([application], columns=FEATURE_COLUMNS)
    return float(model.predict_proba(row)[:, 1][0])


def load_similar_cases_index(index_path: str | Path):
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Similar-cases index not found: {path}. Run train_model.py first.")
    return joblib.load(path)


def search_similar_cases(index_artifact: dict, application: dict, top_k: int = 5) -> list[dict]:
    row = pd.DataFrame([application], columns=FEATURE_COLUMNS)
    scaled = index_artifact["scaler"].transform(row)
    distances, indices = index_artifact["nearest_neighbors"].kneighbors(scaled, n_neighbors=top_k)

    results = []
    for distance, idx in zip(distances[0], indices[0], strict=False):
        similarity = 1 - float(distance)
        results.append(
            {
                "customer_id": f"C{index_artifact['ids'].iloc[idx]}",
                "similarity": round(similarity, 4),
                "actual_default": int(index_artifact["labels"].iloc[idx]),
            }
        )
    return results
