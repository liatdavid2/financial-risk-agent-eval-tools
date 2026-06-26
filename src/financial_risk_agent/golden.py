from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from financial_risk_agent.config import FEATURE_COLUMNS, LANGGRAPH_TOOL_SEQUENCE, TARGET_COLUMN
from financial_risk_agent.data import load_credit_dataset
from financial_risk_agent.rules import expected_decision_from_label_and_rules, payment_behavior_rules

FORBIDDEN_CLAIMS = [
    "fraud",
    "criminal",
    "illegal activity",
    "money laundering",
    "guaranteed approval",
]


def build_golden_dataset(csv_path: str | Path, output_path: str | Path, limit: int = 300) -> int:
    df = load_credit_dataset(csv_path)

    # Balanced-ish sample: include positives and negatives so evaluation is meaningful.
    positives = df[df[TARGET_COLUMN] == 1].head(limit // 2)
    negatives = df[df[TARGET_COLUMN] == 0].head(limit - len(positives))
    sample = (
        pd.concat([positives, negatives], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for _, row in sample.iterrows():
            application = {col: float(row[col]) for col in FEATURE_COLUMNS}
            rules = payment_behavior_rules(application)
            label = int(row[TARGET_COLUMN])
            expected_decision = expected_decision_from_label_and_rules(
                label=label,
                rule_reasons=rules["rule_reasons"],
            )
            record = {
                "customer_id": f"C{row['ID']}",
                "input": {"customer_id": f"C{row['ID']}", **application},
                "actual_default_label": label,
                "expected_decision": expected_decision,
                "expected_reasons": rules["rule_reasons"],
                "expected_tools": LANGGRAPH_TOOL_SEQUENCE,
                "forbidden_claims": FORBIDDEN_CLAIMS,
            }
            f.write(json.dumps(record) + "\n")

    return len(sample)
