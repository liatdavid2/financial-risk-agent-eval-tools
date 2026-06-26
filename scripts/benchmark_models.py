from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_risk_agent.config import (  # noqa: E402
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    RISK_MODEL_PATH,
    SIMILAR_CASES_INDEX_PATH,
    TARGET_COLUMN,
)
from financial_risk_agent.data import load_credit_dataset  # noqa: E402


def build_models(pos_weight: float) -> dict[str, Any]:
    models: dict[str, Any] = {}

    models["random_forest"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1,
        )
    except Exception as exc:
        print(f"Skipping xgboost: {exc}")

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=450,
            max_depth=-1,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as exc:
        print(f"Skipping lightgbm: {exc}")

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=450,
            depth=5,
            learning_rate=0.035,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
        )
    except Exception as exc:
        print(f"Skipping catboost: {exc}")

    return models


def evaluate_model(name: str, model: Any, X_train, X_test, y_train, y_test) -> dict[str, Any]:
    start = time.perf_counter()

    model.fit(X_train, y_train)

    train_seconds = time.perf_counter() - start

    proba = model.predict_proba(X_test)[:, 1]

    # Threshold used only for model benchmark classification report.
    # The API policy still decides approve/manual_review/reject separately.
    pred = (proba >= 0.5).astype(int)

    return {
        "model_name": name,
        "train_seconds": round(train_seconds, 3),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision_pr_auc": float(average_precision_score(y_test, proba)),
        "precision_class_1": float(precision_score(y_test, pred, zero_division=0)),
        "recall_class_1": float(recall_score(y_test, pred, zero_division=0)),
        "f1_class_1": float(f1_score(y_test, pred, zero_division=0)),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }


def save_similar_cases_index(df: pd.DataFrame, index_path: str | Path) -> None:
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark several credit-risk models and save the best one.")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"))
    parser.add_argument("--model-path", default=str(RISK_MODEL_PATH))
    parser.add_argument("--index-path", default=str(SIMILAR_CASES_INDEX_PATH))
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "reports" / "model_benchmark.json"),
    )
    parser.add_argument(
        "--metric",
        default="average_precision_pr_auc",
        choices=["average_precision_pr_auc", "roc_auc", "f1_class_1", "recall_class_1"],
    )
    args = parser.parse_args()

    df = load_credit_dataset(args.csv)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())
    pos_weight = negative_count / max(positive_count, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models = build_models(pos_weight=pos_weight)

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        trained_models[name] = model
        print(
            f"{name}: PR-AUC={metrics['average_precision_pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"F1(class 1)={metrics['f1_class_1']:.4f}"
        )

    results_sorted = sorted(results, key=lambda row: row[args.metric], reverse=True)
    best = results_sorted[0]
    best_name = best["model_name"]
    best_model = trained_models[best_name]

    model_path = Path(args.model_path)
    report_path = Path(args.report)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_path)
    save_similar_cases_index(df, args.index_path)

    report = {
        "selected_metric": args.metric,
        "best_model": best_name,
        "best_score": best[args.metric],
        "rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "results": results_sorted,
        "feature_columns": FEATURE_COLUMNS,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Model benchmark summary ===")
    print(json.dumps(report, indent=2))

    print(f"\nSaved best model: {model_path}")
    print(f"Saved similar-cases index: {Path(args.index_path)}")
    print(f"Saved benchmark report: {report_path}")


if __name__ == "__main__":
    main()