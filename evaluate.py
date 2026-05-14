import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from preprocess import prepare_data
from config import MODEL_DIR, OUTPUT_DIR


def evaluate_model(model_name: str):
    data = prepare_data()
    X_test = data["X_test"]
    y_test = data["y_test"].reset_index(drop=True)

    model_path = MODEL_DIR / f"{model_name}.joblib"
    clf = joblib.load(model_path)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"]
    )
    cm_df.to_csv(OUTPUT_DIR / f"{model_name}_confusion_matrix.csv")

    with open(OUTPUT_DIR / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(OUTPUT_DIR / f"{model_name}_classification_report.csv")

    print(f"\n=== {model_name} ===")
    for k, v in metrics.items():
        if k != "model_name":
            print(f"{k}: {v:.4f}")

    extract_feature_importance(clf, model_name)

    return metrics


def extract_feature_importance(clf, model_name: str):
    preprocessor = clf.named_steps["preprocessor"]
    model = clf.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        imp_type = "abs_coef"
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_type = "feature_importance"
    else:
        print(f"No feature importance available for {model_name}")
        return

    imp_df = pd.DataFrame({
        "feature": feature_names,
        imp_type: importances
    }).sort_values(by=imp_type, ascending=False)

    imp_df.to_csv(OUTPUT_DIR / f"{model_name}_feature_importance.csv", index=False)
    print(f"Saved feature importance for {model_name}")


if __name__ == "__main__":
    all_metrics = []
    for model_name in ["logistic_regression", "random_forest"]:
        metrics = evaluate_model(model_name)
        all_metrics.append(metrics)

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nSaved model comparison.")