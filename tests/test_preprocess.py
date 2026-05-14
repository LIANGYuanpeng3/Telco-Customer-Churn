"""数据预处理与推理前处理的基础回归测试。"""

import joblib
import pandas as pd
import pytest

from config import ID_COL, MODEL_DIR, TARGET_COL, TEST_SIZE
from preprocess import align_inference_to_fitted_pipeline, prepare_data, prepare_inference_data

# 与 API 输入一致的一行模拟顾客（原始取值，经 prepare_inference_data 清洗）
SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 53.85,
    "TotalCharges": 468.35,
}


def test_prepare_data_splits_nonempty():
    data = prepare_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    assert len(X_train) > 0 and len(X_test) > 0
    assert len(y_train) > 0 and len(y_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_prepare_data_no_customer_id_or_target_in_features():
    data = prepare_data()
    for name in ("X_train", "X_test"):
        df = data[name]
        assert ID_COL not in df.columns, f"{name} 不应包含 {ID_COL}"
        assert TARGET_COL not in df.columns, f"{name} 不应包含目标列 {TARGET_COL}"


def test_prepare_data_train_test_ratio_reasonable():
    data = prepare_data()
    n = len(data["X_train"]) + len(data["X_test"])
    ratio = len(data["X_test"]) / n
    assert ratio == pytest.approx(TEST_SIZE, abs=0.02)


def test_prepare_inference_data_single_row_and_pipeline_accepts():
    raw = pd.DataFrame([SAMPLE_CUSTOMER])
    out = prepare_inference_data(raw)
    assert not out.empty
    assert len(out) == 1

    model_path = MODEL_DIR / "random_forest.joblib"
    if not model_path.is_file():
        pytest.skip("未找到 models/random_forest.joblib，跳过 Pipeline 兼容性断言")

    try:
        clf = joblib.load(model_path)
    except Exception as exc:
        pytest.skip(f"模型无法加载（sklearn 版本等）: {exc}")

    aligned = align_inference_to_fitted_pipeline(out, clf)
    probs = clf.predict_proba(aligned)
    assert probs.shape == (1, 2)
