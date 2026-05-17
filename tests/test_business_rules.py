import pandas as pd

from business_rules import assign_risk_level, expected_cost, suggest_action


def test_assign_risk_level_tiers():
    assert assign_risk_level(0.85) == "High"
    assert assign_risk_level(0.55) == "Medium"
    assert assign_risk_level(0.10) == "Low"


def test_suggest_action_high_tenure():
    row = pd.Series({"risk_level": "High", "tenure": 6, "MonthlyCharges": 50})
    assert "onboarding" in suggest_action(row).lower() or "new-customer" in suggest_action(row).lower()


def test_expected_cost_increases_with_fn():
    cfg = {
        "cost_false_negative": 500,
        "cost_false_positive": 15,
        "cost_outreach": 15,
        "saved_customer_ltv": 400,
        "retention_success_rate": 0.35,
    }
    low_fn = expected_cost(10, 10, 1, 100, cfg)
    high_fn = expected_cost(10, 10, 50, 100, cfg)
    assert high_fn > low_fn
