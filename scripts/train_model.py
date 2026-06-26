from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_risk_agent.config import PROJECT_ROOT, RISK_MODEL_PATH, SIMILAR_CASES_INDEX_PATH
from financial_risk_agent.modeling import train_credit_risk_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit-risk model and similar-cases index.")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"))
    parser.add_argument("--model-path", default=str(RISK_MODEL_PATH))
    parser.add_argument("--index-path", default=str(SIMILAR_CASES_INDEX_PATH))
    parser.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "training_metrics.json"))
    args = parser.parse_args()

    metrics = train_credit_risk_model(
        csv_path=args.csv,
        model_path=args.model_path,
        index_path=args.index_path,
        reports_path=args.report,
    )
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {Path(args.model_path)}")
    print(f"Saved similar-cases index: {Path(args.index_path)}")
    print(f"Saved report: {Path(args.report)}")


if __name__ == "__main__":
    main()
