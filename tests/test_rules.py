from financial_risk_agent.rules import payment_behavior_rules


def test_payment_behavior_rules_detect_risk_signals():
    app = {
        "LIMIT_BAL": 20000,
        "PAY_0": 3,
        "PAY_2": 2,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 18500,
        "PAY_AMT1": 400,
    }

    result = payment_behavior_rules(app)

    assert "recent_payment_delay" in result["rule_reasons"]
    assert "repeated_payment_delay" in result["rule_reasons"]
    assert "high_credit_utilization" in result["rule_reasons"]
    assert "low_recent_payment" in result["rule_reasons"]
