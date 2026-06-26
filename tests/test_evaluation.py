from financial_risk_agent.eval_metrics import (
    groundedness_score,
    reason_coverage,
    tool_calling_completeness,
    tool_order_match,
)


def test_reason_coverage():
    assert reason_coverage(["a", "b"], ["a", "c"]) == 0.5
    assert reason_coverage([], []) == 1.0


def test_tool_calling_metrics():
    expected = ["normalize_application", "credit_risk_model_predict"]
    actual = ["normalize_application", "credit_risk_model_predict"]
    assert tool_calling_completeness(expected, actual) == 1.0
    assert tool_order_match(expected, actual)


def test_groundedness_score():
    text = "The main evidence is: recent repayment delay."
    assert groundedness_score(["recent_payment_delay"], text) == 1.0
