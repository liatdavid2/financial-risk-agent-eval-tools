from __future__ import annotations

from financial_risk_agent.rules import REASON_TEXT


def generate_explanation(
    decision: str,
    risk_level: str,
    default_probability: float,
    reasons: list[str],
    similar_cases: list[dict],
) -> str:
    """Generate a grounded explanation.

    In production, this could be replaced with a heavy LLM call. The important pattern is that
    the explanation receives only tool outputs: policy, ML score, rule evidence, and similar cases.
    Redis caches it to reduce repeated LLM cost and latency.
    """
    if reasons:
        reason_text = ", ".join(REASON_TEXT.get(reason, reason) for reason in reasons)
    else:
        reason_text = "no major payment behavior risk signals were triggered"

    similar_default_count = sum(case.get("actual_default", 0) for case in similar_cases)
    similar_sentence = (
        f" Among the retrieved similar historical cases, {similar_default_count} had a default label."
        if similar_cases
        else " No similar historical cases were available."
    )

    if decision == "approve":
        action = "The application can be approved"
    elif decision == "reject":
        action = "The application should be rejected or escalated according to policy"
    else:
        action = "The application should be sent to manual review"

    return (
        f"{action}. The estimated default probability is {default_probability:.2%}, "
        f"with overall risk level '{risk_level}'. The main evidence is: {reason_text}."
        f"{similar_sentence}"
    )
