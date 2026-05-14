import joblib
import pandas as pd

from preprocess import prepare_data
from config import MODEL_DIR, OUTPUT_DIR, load_decision_threshold


def assign_risk_level(prob):
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    return "Low"


def suggest_action(row):
    if row["risk_level"] == "High" and row["tenure"] < 12:
        return "Offer onboarding support or new-customer retention campaign"
    if row["risk_level"] == "High" and row["MonthlyCharges"] >= 70:
        return "Provide targeted discount or plan optimization"
    if row["risk_level"] == "High":
        return "Prioritize retention outreach"
    if row["risk_level"] == "Medium":
        return "Monitor behavior and send personalized campaign"
    return "Maintain regular engagement"


def main(model_name="random_forest"):
    data = prepare_data()

    X_test = data["X_test"].copy().reset_index(drop=True)
    y_test = data["y_test"].reset_index(drop=True)
    id_test = data["id_test"].reset_index(drop=True)

    clf = joblib.load(MODEL_DIR / f"{model_name}.joblib")

    # 解约业务常用「概率 + 可调阈值」判定正类，避免固定 0.5 漏检高成本流失客户（见 threshold_analysis / README）
    threshold = load_decision_threshold(model_name)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    result_df = pd.DataFrame({
        "customerID": id_test,
        "actual_churn": y_test,
        "predicted_churn": pred,
        "churn_probability": prob,
    })

    # add selected raw/business columns
    for col in ["tenure", "MonthlyCharges", "Contract", "InternetService", "PaymentMethod"]:
        if col in X_test.columns:
            result_df[col] = X_test[col].values

    result_df["risk_level"] = result_df["churn_probability"].apply(assign_risk_level)
    result_df["suggested_action"] = result_df.apply(suggest_action, axis=1)

    result_df = result_df.sort_values(by="churn_probability", ascending=False)

    result_df.to_csv(OUTPUT_DIR / f"{model_name}_predictions.csv", index=False)

    high_risk_df = result_df[result_df["risk_level"] == "High"].copy()
    high_risk_df.to_csv(OUTPUT_DIR / f"{model_name}_high_risk_customers.csv", index=False)

    print(f"Decision threshold (prob >= t): {threshold}")
    print(f"Saved full predictions: {OUTPUT_DIR / f'{model_name}_predictions.csv'}")
    print(f"Saved high-risk customers: {OUTPUT_DIR / f'{model_name}_high_risk_customers.csv'}")


if __name__ == "__main__":
    main()