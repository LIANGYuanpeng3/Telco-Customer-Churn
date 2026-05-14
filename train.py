import joblib
import json
from datetime import datetime, timezone

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess import prepare_data
from config import MODEL_DIR


def train_models():
    data = prepare_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    preprocessor = data["preprocessor"]

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
    }

    trained = {}

    for name, model in models.items():
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        clf.fit(X_train, y_train)

        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(clf, model_path)

        trained[name] = str(model_path)
        print(f"Saved: {model_path}")

    meta = {
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_models": trained,
    }
    with open(MODEL_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    train_models()