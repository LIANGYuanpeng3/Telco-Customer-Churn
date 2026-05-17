from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from business_rules import assign_risk_level, suggest_action
from config import load_decision_threshold, load_training_meta
from preprocess import align_inference_to_fitted_pipeline, prepare_inference_data

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "random_forest"
MODEL_PATH = BASE_DIR / "models" / f"{MODEL_NAME}.joblib"

app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")

model: Any = None
MODEL_LOAD_ERROR: str | None = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    MODEL_LOAD_ERROR = f"{type(e).__name__}: {e}"

DECISION_THRESHOLD = load_decision_threshold(MODEL_NAME)


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class BatchPredictRequest(BaseModel):
    customers: list[CustomerInput] = Field(
        ...,
        description="待预测客户列表，至少 1 条",
        min_length=1,
    )


def dataframe_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    推理路径：必须先 prepare_inference_data（与训练共享清洗/特征），再按拟合 Pipeline 对齐列，避免 train-serving skew。
    """
    df = prepare_inference_data(df)
    return align_inference_to_fitted_pipeline(df, model)


def prediction_payload(prob: float, row: pd.Series) -> dict:
    return {
        "churn_probability": round(float(prob), 4),
        "predicted_churn": int(prob >= DECISION_THRESHOLD),
        "risk_level": assign_risk_level(prob),
        "recommended_action": suggest_action(row, prob),
    }


def require_model() -> None:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "message": "模型文件未能加载，推理不可用。请检查 models/random_forest.joblib 与 sklearn 版本是否与训练环境一致。",
                "detail": MODEL_LOAD_ERROR,
            },
        )


def feature_list_for_model() -> list[str]:
    if model is None:
        return []
    pre = model.named_steps.get("preprocessor")
    if pre is None or not hasattr(pre, "feature_names_in_"):
        return []
    return [str(x) for x in pre.feature_names_in_]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError):
    """将 Pydantic 校验失败整理为稳定结构，便于调用方定位缺字段或类型错误。"""
    errors = []
    for err in exc.errors():
        errors.append(
            {
                "field": ".".join(str(loc) for loc in err.get("loc", ())),
                "message": err.get("msg"),
                "type": err.get("type"),
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "请求体字段缺失或类型不正确，请对照 CustomerInput 模型检查。",
            "errors": errors,
        },
    )


@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API is running."}


@app.get("/health")
def health():
    """存活探针：服务是否可用、模型是否已加载。"""
    loaded = model is not None
    st = "healthy" if loaded else "degraded"
    return {
        "status": st,
        "service_status": st,
        "model_loaded": loaded,
        "model_name": MODEL_NAME,
        **({"model_load_error": MODEL_LOAD_ERROR} if not loaded else {}),
    }


@app.get("/model_info")
def model_info():
    """模型元数据：名称、版本/训练时间（training_meta.json）、业务阈值、输入特征列顺序。"""
    meta = load_training_meta()
    return {
        "model_name": MODEL_NAME,
        "model_version": meta.get("model_version", "not_set"),
        "selected_threshold": DECISION_THRESHOLD,
        "feature_list": feature_list_for_model(),
        "training_date": meta.get("trained_at") or meta.get("training_date"),
        "model_loaded": model is not None,
        **({"model_load_error": MODEL_LOAD_ERROR} if model is None else {}),
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    require_model()
    try:
        raw = pd.DataFrame([customer.model_dump()])
        df = dataframe_for_model(raw)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "inference_preparation_failed",
                "message": str(e),
            },
        ) from e

    prob = float(model.predict_proba(df)[0, 1])
    return prediction_payload(prob, df.iloc[0])


@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest):
    require_model()
    try:
        raw = pd.DataFrame([c.model_dump() for c in req.customers])
        df = dataframe_for_model(raw)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "inference_preparation_failed",
                "message": str(e),
            },
        ) from e

    probs = model.predict_proba(df)[:, 1]
    results = [prediction_payload(float(probs[i]), df.iloc[i]) for i in range(len(df))]
    return {"results": results}
