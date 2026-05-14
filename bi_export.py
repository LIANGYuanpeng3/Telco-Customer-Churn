"""
BI 导出：将测试集预测整理为扁平 CSV，便于 Power BI / Tableau / Looker Studio / Streamlit 直连，
把「模型输出」转为「分群、收入与施策」视角的决策支持表。
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import MODEL_DIR, OUTPUT_DIR, load_decision_threshold
from export_results import assign_risk_level, suggest_action
from preprocess import prepare_data

OUTPUT_BI = OUTPUT_DIR / "bi"

CUSTOMER_PREDICTION_COLUMNS = [
    "customerID",
    "churn_probability",
    "predicted_churn",
    "actual_churn",
    "risk_level",
    "recommended_action",
    "MonthlyCharges",
    "TotalCharges",
    "tenure",
    "Contract",
    "PaymentMethod",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
]


def tenure_group_labels(ser: pd.Series) -> pd.Series:
    """0-6 / 7-12 / 13-24 / 25-48 / 49+，与 BI 分桶报表对齐。"""
    return pd.cut(
        ser.astype(float),
        bins=[0, 7, 13, 25, 49, np.inf],
        right=False,
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    ).astype(str)


def build_customer_level_df(model_name: str) -> tuple[pd.DataFrame, float]:
    data = prepare_data()
    X_test = data["X_test"].copy().reset_index(drop=True)
    y_test = data["y_test"].reset_index(drop=True)
    id_test = data["id_test"].reset_index(drop=True)

    clf = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    threshold = load_decision_threshold(model_name)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "customerID": id_test.astype(str),
            "churn_probability": prob.astype(float),
            "predicted_churn": pred.astype(int),
            "actual_churn": y_test.astype(int),
        }
    )

    for col in CUSTOMER_PREDICTION_COLUMNS:
        if col in ("customerID", "churn_probability", "predicted_churn"):
            continue
        if col == "risk_level" or col == "recommended_action":
            continue
        if col in X_test.columns:
            df[col] = X_test[col].values
        else:
            df[col] = np.nan

    df["risk_level"] = df["churn_probability"].apply(assign_risk_level)
    df["recommended_action"] = df.apply(suggest_action, axis=1)

    return df, threshold


def write_customer_predictions(df: pd.DataFrame) -> None:
    out = df[CUSTOMER_PREDICTION_COLUMNS].copy()
    out.to_csv(OUTPUT_BI / "customer_predictions.csv", index=False)


def write_risk_segment_summary(df: pd.DataFrame) -> None:
    g = df.groupby("risk_level", dropna=False)
    summary = g.agg(
        customer_count=("customerID", "count"),
        average_churn_probability=("churn_probability", "mean"),
        total_monthly_revenue=("MonthlyCharges", "sum"),
        average_monthly_charges=("MonthlyCharges", "mean"),
    ).reset_index()
    order = pd.CategoricalDtype(["High", "Medium", "Low"], ordered=True)
    summary["risk_level"] = summary["risk_level"].astype(order)
    summary = summary.sort_values("risk_level")
    summary.to_csv(OUTPUT_BI / "risk_segment_summary.csv", index=False)


def write_churn_by_contract(df: pd.DataFrame) -> None:
    base = df.groupby("Contract", dropna=False).agg(
        customer_count=("customerID", "count"),
        actual_churn_rate=("actual_churn", "mean"),
        average_predicted_churn_probability=("churn_probability", "mean"),
    ).reset_index()

    high = (
        df.loc[df["risk_level"] == "High"]
        .groupby("Contract", dropna=False)
        .size()
        .reset_index(name="high_risk_customer_count")
    )
    summary = base.merge(high, on="Contract", how="left")
    summary["high_risk_customer_count"] = (
        summary["high_risk_customer_count"].fillna(0).astype(int)
    )
    summary.to_csv(OUTPUT_BI / "churn_by_contract.csv", index=False)


def write_churn_by_tenure_group(df: pd.DataFrame) -> None:
    d = df.copy()
    d["tenure_group"] = tenure_group_labels(d["tenure"])
    g = d.groupby("tenure_group", dropna=False)
    summary = g.agg(
        customer_count=("customerID", "count"),
        actual_churn_rate=("actual_churn", "mean"),
        average_predicted_churn_probability=("churn_probability", "mean"),
    ).reset_index()
    tg_order = pd.CategoricalDtype(["0-6", "7-12", "13-24", "25-48", "49+"], ordered=True)
    summary["tenure_group"] = summary["tenure_group"].astype(tg_order)
    summary = summary.sort_values("tenure_group")
    summary.to_csv(OUTPUT_BI / "churn_by_tenure_group.csv", index=False)


def write_retention_action_summary(df: pd.DataFrame) -> None:
    g = df.groupby("recommended_action", dropna=False)
    summary = g.agg(
        customer_count=("customerID", "count"),
        total_monthly_revenue=("MonthlyCharges", "sum"),
        average_churn_probability=("churn_probability", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT_BI / "retention_action_summary.csv", index=False)


def write_model_kpi_summary(
    model_name: str, threshold: float, y_true: pd.Series, y_prob: np.ndarray, y_pred: np.ndarray
) -> None:
    row = {
        "model_name": model_name,
        "selected_threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    pd.DataFrame([row]).to_csv(OUTPUT_BI / "model_kpi_summary.csv", index=False)


def main(model_name: str = "random_forest") -> None:
    OUTPUT_BI.mkdir(parents=True, exist_ok=True)

    df, threshold = build_customer_level_df(model_name)

    write_customer_predictions(df)
    write_risk_segment_summary(df)
    write_churn_by_contract(df)
    write_churn_by_tenure_group(df)
    write_retention_action_summary(df)
    write_model_kpi_summary(
        model_name,
        threshold,
        df["actual_churn"],
        df["churn_probability"].values,
        df["predicted_churn"].values,
    )

    print(f"BI CSV files written to {OUTPUT_BI}")


if __name__ == "__main__":
    main()
