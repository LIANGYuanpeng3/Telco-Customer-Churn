import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_PATH, TARGET_COL, ID_COL, TEST_SIZE, RANDOM_STATE


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # 推理经 prepare_inference_data() 复用本函数；勿在 API 重复映射，以免 train-serving skew。
    df = df.copy()

    # TotalCharges may contain blanks
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # target: Yes/No -> 1/0, only when target column exists
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # binary columns
    binary_map = {"Yes": 1, "No": 0}
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # 推理经 prepare_inference_data() 复用本函数；特征定义必须与训练完全一致。
    df = df.copy()

    tenure_safe = df["tenure"].replace(0, np.nan)

    df["avg_monthly_spend"] = df["TotalCharges"] / tenure_safe
    df["avg_monthly_spend"] = df["avg_monthly_spend"].replace([np.inf, -np.inf], np.nan)

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    existing_service_cols = [c for c in service_cols if c in df.columns]

    def count_services(row):
        cnt = 0
        for c in existing_service_cols:
            if row[c] == "Yes":
                cnt += 1
        return cnt

    df["service_count"] = df.apply(count_services, axis=1)
    df["is_new_customer"] = (df["tenure"] < 12).astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    return df


def prepare_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    推理阶段与训练阶段共用的「进入 ColumnTransformer 之前」的表。
    必须复用 basic_cleaning + feature_engineering，避免在 API 另写一套逻辑造成 train-serving skew。
    """
    df = df.copy()
    df = basic_cleaning(df)
    df = feature_engineering(df)
    # 与 prepare_data() 中 X_train_model 一致：建模特征中不含 ID，也不应携带标签列
    drop_cols = [c for c in (TARGET_COL, ID_COL) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def align_inference_to_fitted_pipeline(
    df: pd.DataFrame, fitted_pipeline: SklearnPipeline
) -> pd.DataFrame:
    """
    按拟合后 ColumnTransformer 记录的 feature_names_in_ 截断并重排列顺序，
    与训练时传入 fit 的 DataFrame 列布局一致，避免列顺序漂移或多余列带来的 train-serving skew。
    """
    if "preprocessor" not in fitted_pipeline.named_steps:
        raise ValueError("fitted_pipeline 中缺少名为 'preprocessor' 的步骤")
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError(
            "预处理器缺少 feature_names_in_；请使用在 DataFrame 上拟合的 Pipeline，"
            "或升级 sklearn 后重新训练并保存模型。"
        )
    expected = list(preprocessor.feature_names_in_)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"推理数据缺少训练期特征列: {missing}")
    # 只保留训练期列，且顺序与 fit 时一致
    return df.loc[:, expected].copy()


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    if ID_COL in X.columns:
        customer_ids = X[ID_COL].copy()
    else:
        customer_ids = pd.Series(np.arange(len(X)), name="customerID")

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        customer_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, id_train, id_test


def build_preprocessor(X: pd.DataFrame):
    X = X.copy()

    feature_cols = [c for c in X.columns if c != ID_COL]
    X_model = X[feature_cols]

    numeric_features = X_model.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X_model.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_transformer = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, feature_cols, numeric_features, categorical_features


def prepare_data():
    df = load_data()
    df = basic_cleaning(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test, id_train, id_test = split_features_target(df)
    preprocessor, feature_cols, numeric_features, categorical_features = build_preprocessor(X_train)

    X_train_model = X_train[feature_cols].copy()
    X_test_model = X_test[feature_cols].copy()

    return {
        "raw_df": df,
        "X_train": X_train_model,
        "X_test": X_test_model,
        "y_train": y_train,
        "y_test": y_test,
        "id_train": id_train.reset_index(drop=True),
        "id_test": id_test.reset_index(drop=True),
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }