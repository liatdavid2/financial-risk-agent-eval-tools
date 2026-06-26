from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_risk_agent.agent import investigate_application

SAMPLE = {
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


def main() -> None:
    result = investigate_application(SAMPLE)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
