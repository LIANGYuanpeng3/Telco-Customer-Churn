"""
Shared business logic: risk tiers, retention actions, and cost-benefit simulation.
Configs live under config/ so rules can change without code edits.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"

DEFAULT_COST_CONFIG = {
    "cost_false_negative": 500.0,
    "cost_false_positive": 15.0,
    "cost_outreach": 15.0,
    "saved_customer_ltv": 400.0,
    "retention_success_rate": 0.35,
}

DEFAULT_RISK_TIERS = [
    {"name": "High", "min_probability": 0.70},
    {"name": "Medium", "min_probability": 0.40},
    {"name": "Low", "min_probability": 0.0},
]

DEFAULT_RETENTION_RULES = {
    "default_action": "Maintain regular engagement",
    "rules": [
        {
            "risk_levels": ["High"],
            "conditions": {"tenure": {"lt": 12}},
            "action": "Offer onboarding support or new-customer retention campaign",
        },
        {
            "risk_levels": ["High"],
            "conditions": {"MonthlyCharges": {"gte": 70}},
            "action": "Provide targeted discount or plan optimization",
        },
        {
            "risk_levels": ["High"],
            "conditions": {},
            "action": "Prioritize retention outreach",
        },
        {
            "risk_levels": ["Medium"],
            "conditions": {},
            "action": "Monitor behavior and send personalized campaign",
        },
    ],
}


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.is_file():
        return fallback
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return fallback


def load_cost_config() -> dict[str, float]:
    raw = _load_json(CONFIG_DIR / "cost_config.json", DEFAULT_COST_CONFIG)
    return {
        "cost_false_negative": float(raw.get("cost_false_negative", 500)),
        "cost_false_positive": float(raw.get("cost_false_positive", 15)),
        "cost_outreach": float(raw.get("cost_outreach", 15)),
        "saved_customer_ltv": float(raw.get("saved_customer_ltv", 400)),
        "retention_success_rate": float(raw.get("retention_success_rate", 0.35)),
    }


def load_risk_tiers() -> list[dict[str, Any]]:
    raw = _load_json(CONFIG_DIR / "risk_tiers.json", {"tiers": DEFAULT_RISK_TIERS})
    tiers = raw.get("tiers", DEFAULT_RISK_TIERS)
    return sorted(tiers, key=lambda t: float(t["min_probability"]), reverse=True)


def load_retention_rules() -> dict[str, Any]:
    return _load_json(CONFIG_DIR / "retention_rules.json", DEFAULT_RETENTION_RULES)


def assign_risk_level(prob: float) -> str:
    for tier in load_risk_tiers():
        if prob >= float(tier["min_probability"]):
            return str(tier["name"])
    return "Low"


def _condition_matches(row: pd.Series, conditions: dict[str, dict[str, float]]) -> bool:
    if not conditions:
        return True
    for field, spec in conditions.items():
        val = pd.to_numeric(row.get(field), errors="coerce")
        if spec.get("lt") is not None and not (pd.notna(val) and val < spec["lt"]):
            return False
        if spec.get("lte") is not None and not (pd.notna(val) and val <= spec["lte"]):
            return False
        if spec.get("gt") is not None and not (pd.notna(val) and val > spec["gt"]):
            return False
        if spec.get("gte") is not None and not (pd.notna(val) and val >= spec["gte"]):
            return False
    return True


def suggest_action(row: pd.Series, prob: float | None = None) -> str:
    risk_level = row.get("risk_level")
    if risk_level is None or (isinstance(risk_level, float) and pd.isna(risk_level)):
        risk_level = assign_risk_level(prob if prob is not None else 0.0)
    risk_level = str(risk_level)

    cfg = load_retention_rules()
    for rule in cfg.get("rules", []):
        if risk_level not in rule.get("risk_levels", []):
            continue
        if _condition_matches(row, rule.get("conditions", {})):
            return str(rule["action"])
    return str(cfg.get("default_action", "Maintain regular engagement"))


def expected_cost(tp: int, fp: int, fn: int, tn: int, cfg: dict[str, float] | None = None) -> float:
    """Expected portfolio cost for a confusion matrix at one threshold (illustrative)."""
    c = cfg or load_cost_config()
    outreach = c["cost_outreach"]
    saved = c["retention_success_rate"] * c["saved_customer_ltv"]
    return float(
        fn * c["cost_false_negative"]
        + fp * c["cost_false_positive"]
        + tp * (outreach - saved)
    )


def recommend_threshold_by_cost(df: pd.DataFrame) -> tuple[float, str]:
    row = df.loc[df["expected_cost"].idxmin()]
    return float(row["threshold"]), "Minimize expected_cost (config/cost_config.json)"


def publish_data_mart_manifest(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = CONFIG_DIR / "data_mart_manifest.json"
    if src.is_file():
        shutil.copy2(src, dest_dir / "data_mart_manifest.json")
