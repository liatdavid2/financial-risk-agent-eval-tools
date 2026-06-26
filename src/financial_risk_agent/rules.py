from __future__ import annotations


REASON_TEXT = {
    "recent_payment_delay": "recent repayment delay",
    "repeated_payment_delay": "repeated repayment delays across multiple months",
    "high_credit_utilization": "high utilization of the credit limit",
    "low_recent_payment": "low recent payment compared with the latest bill amount",
    "bill_above_credit_limit": "latest bill amount above the credit limit",
}


def payment_behavior_rules(application: dict) -> dict:
    """Transparent financial risk rules.

    These are intentionally simple and readable so they can be used both by the agent and by
    the golden dataset builder. This makes GenAI evaluation deterministic and explainable.
    """
    reasons: list[str] = []
    evidence: dict[str, float | int | str] = {}

    limit_bal = float(application.get("LIMIT_BAL", 0) or 0)
    bill_amt1 = float(application.get("BILL_AMT1", 0) or 0)
    pay_amt1 = float(application.get("PAY_AMT1", 0) or 0)
    pay_0 = float(application.get("PAY_0", 0) or 0)
    past_payments = [
        float(application.get(col, 0) or 0) for col in ["PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    ]

    utilization = bill_amt1 / limit_bal if limit_bal > 0 else 0.0
    delayed_months = int(sum(1 for value in [pay_0, *past_payments] if value >= 2))

    if pay_0 >= 2:
        reasons.append("recent_payment_delay")
        evidence["PAY_0"] = pay_0

    if delayed_months >= 2:
        reasons.append("repeated_payment_delay")
        evidence["delayed_months"] = delayed_months

    if utilization >= 0.8:
        reasons.append("high_credit_utilization")
        evidence["credit_utilization"] = round(utilization, 3)

    if bill_amt1 > 0 and pay_amt1 < bill_amt1 * 0.05:
        reasons.append("low_recent_payment")
        evidence["payment_to_bill_ratio"] = round(pay_amt1 / bill_amt1, 3)

    if limit_bal > 0 and bill_amt1 > limit_bal:
        reasons.append("bill_above_credit_limit")
        evidence["bill_to_limit_ratio"] = round(bill_amt1 / limit_bal, 3)

    return {
        "rule_reasons": reasons,
        "rule_evidence": evidence,
        "rule_risk_level": "high" if len(reasons) >= 3 else "medium" if reasons else "low",
    }


def expected_decision_from_label_and_rules(label: int, rule_reasons: list[str]) -> str:
    """Build deterministic golden labels for GenAI evaluation.

    The UCI label means default next month. In a real financial workflow, even risky cases
    should normally become manual review rather than unsupported claims like fraud/criminal.
    """
    severe = {"recent_payment_delay", "repeated_payment_delay", "bill_above_credit_limit"}
    severe_count = sum(1 for reason in rule_reasons if reason in severe)

    if label == 1 or severe_count >= 2 or len(rule_reasons) >= 3:
        return "manual_review"
    return "approve"
