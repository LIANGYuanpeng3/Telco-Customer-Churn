import joblib
import pandas as pd

from business_rules import assign_risk_level, suggest_action
from preprocess import prepare_data
from config import MODEL_DIR, OUTPUT_DIR, load_decision_threshold


def main(model_name="random_forest"):
    data = prepare_data()

    X_test = data["X_test"].copy().reset_index(drop=True)
    y_test = data["y_test"].reset_index(drop=True)
    id_test = data["id_test"].reset_index(drop=True)

    clf = joblib.load(MODEL_DIR / f"{model_name}.joblib")

    threshold = load_decision_threshold(model_name)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    result_df = pd.DataFrame({
        "customerID": id_test,
        "actual_churn": y_test,
        "predicted_churn": pred,
        "churn_probability": prob,
    })

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
