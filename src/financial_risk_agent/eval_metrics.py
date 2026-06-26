from __future__ import annotations

from financial_risk_agent.rules import REASON_TEXT


def reason_coverage(expected: list[str], actual: list[str]) -> float:
    if not expected:
        return 1.0
    return len(set(expected) & set(actual)) / len(set(expected))


def contains_forbidden_claim(text: str, forbidden_claims: list[str]) -> bool:
    lower = text.lower()
    return any(claim.lower() in lower for claim in forbidden_claims)


def tool_calling_completeness(expected_tools: list[str], actual_tools: list[str]) -> float:
    if not expected_tools:
        return 1.0
    return len(set(expected_tools) & set(actual_tools)) / len(set(expected_tools))


def tool_order_match(expected_tools: list[str], actual_tools: list[str]) -> bool:
    return actual_tools == expected_tools


def groundedness_score(expected_reasons: list[str], explanation: str) -> float:
    """Simple deterministic groundedness proxy.

    In production this can be replaced by an LLM-as-judge check, but this local metric is useful
    because it is stable, free, and CI-friendly.
    """
    if not expected_reasons:
        return 1.0
    text = explanation.lower()
    hits = 0
    for reason in expected_reasons:
        phrase = REASON_TEXT.get(reason, reason).lower()
        if phrase in text or reason.lower() in text:
            hits += 1
    return hits / len(set(expected_reasons))
