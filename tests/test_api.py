"""FastAPI 基础契约测试（健康检查与单条预测）。"""

import pytest
from fastapi.testclient import TestClient

from app import MODEL_LOAD_ERROR, app, model

SAMPLE_PREDICT_BODY = {
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


@pytest.fixture
def client():
    return TestClient(app)


def test_health_returns_200_with_status_and_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] in ("healthy", "degraded")
    assert isinstance(data["model_loaded"], bool)


@pytest.mark.skipif(model is None, reason=f"模型未加载，跳过 /predict 测试: {MODEL_LOAD_ERROR}")
def test_predict_valid_payload_returns_expected_fields(client):
    r = client.post("/predict", json=SAMPLE_PREDICT_BODY)
    assert r.status_code == 200, r.text
    data = r.json()
    for key in ("churn_probability", "predicted_churn", "risk_level", "recommended_action"):
        assert key in data
    assert isinstance(data["churn_probability"], (int, float))
    assert data["predicted_churn"] in (0, 1)
    assert data["risk_level"] in ("High", "Medium", "Low")
    assert isinstance(data["recommended_action"], str) and len(data["recommended_action"]) > 0
