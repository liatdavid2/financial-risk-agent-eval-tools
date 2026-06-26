from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from financial_risk_agent.agent import investigate_application
from financial_risk_agent.eval_metrics import (
    contains_forbidden_claim,
    groundedness_score,
    reason_coverage,
    tool_calling_completeness,
    tool_order_match,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_golden_eval(golden_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    records = load_jsonl(golden_path)
    if not records:
        raise ValueError(f"No golden records found in {golden_path}")

    rows = []
    started = time.perf_counter()

    for record in records:
        prediction = investigate_application(record["input"])
        actual_reasons = prediction["reasons"]
        explanation = prediction["explanation"]
        expected_tools = record.get("expected_tools", [])
        actual_tools = prediction.get("tool_trace", [])

        row = {
            "customer_id": record["customer_id"],
            "expected_decision": record["expected_decision"],
            "actual_decision": prediction["decision"],
            "decision_match": prediction["decision"] == record["expected_decision"],
            "reason_coverage": reason_coverage(record["expected_reasons"], actual_reasons),
            "groundedness_score": groundedness_score(record["expected_reasons"], explanation),
            "has_forbidden_claim": contains_forbidden_claim(
                explanation, record.get("forbidden_claims", [])
            ),
            "tool_calling_completeness": tool_calling_completeness(expected_tools, actual_tools),
            "tool_order_match": tool_order_match(expected_tools, actual_tools),
            "latency_ms": prediction["latency_ms"],
            "cache": prediction["cache"],
            "expected_reasons": record["expected_reasons"],
            "actual_reasons": actual_reasons,
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
        }
        rows.append(row)

    total = len(rows)
    cache_checks = sum(len(row.get("cache", {})) for row in rows)
    cache_hits = sum(int(value) for row in rows for value in row.get("cache", {}).values())

    result = {
        "total_cases": total,
        "decision_accuracy": sum(row["decision_match"] for row in rows) / total,
        "avg_reason_coverage": sum(row["reason_coverage"] for row in rows) / total,
        "avg_groundedness_score": sum(row["groundedness_score"] for row in rows) / total,
        "hallucination_rate": sum(row["has_forbidden_claim"] for row in rows) / total,
        "avg_tool_calling_completeness": sum(row["tool_calling_completeness"] for row in rows) / total,
        "tool_order_accuracy": sum(row["tool_order_match"] for row in rows) / total,
        "avg_latency_ms": sum(row["latency_ms"] for row in rows) / total,
        "cache_hit_rate": cache_hits / cache_checks if cache_checks else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "rows": rows,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
