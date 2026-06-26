from __future__ import annotations


def apply_policy(default_probability: float, rule_reasons: list[str], similar_cases: list[dict]) -> dict:
    """Convert model score + rule evidence + similar cases into an operational decision."""
    similar_default_rate = 0.0
    if similar_cases:
        similar_default_rate = sum(case["actual_default"] for case in similar_cases) / len(similar_cases)

    evidence_score = len(rule_reasons)

    # This demo is conservative: most risky cases go to manual review.
    # Reject is reserved for very high model score + multiple deterministic signals.
    if default_probability >= 0.78 and evidence_score >= 4:
        decision = "reject"
        risk_level = "very_high"
    elif default_probability >= 0.45 or evidence_score >= 2 or similar_default_rate >= 0.5:
        decision = "manual_review"
        risk_level = "high"
    elif default_probability >= 0.25 or evidence_score == 1:
        decision = "manual_review"
        risk_level = "medium"
    else:
        decision = "approve"
        risk_level = "low"

    return {
        "decision": decision,
        "risk_level": risk_level,
        "similar_default_rate": round(similar_default_rate, 3),
        "policy_version": "credit-risk-policy-v1",
    }
