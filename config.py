import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
THRESHOLD_CONFIG_PATH = MODEL_DIR / "threshold_config.json"
TRAINING_META_PATH = MODEL_DIR / "training_meta.json"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 默认 0.5 与 sklearn 二分类默认决策一致；运行 threshold_analysis.py 后由 threshold_config.json 覆盖
DEFAULT_DECISION_THRESHOLD = 0.5


def load_decision_threshold(model_name: str, default: float = DEFAULT_DECISION_THRESHOLD) -> float:
    """
    读取为业务选定的解约判定阈值（基于概率 >= 阈值 视为正类）。
    解约场景常需低于 0.5 以提高 recall，具体值由 threshold_analysis 在测试集上产生。
    """
    if not THRESHOLD_CONFIG_PATH.is_file():
        return default
    try:
        with open(THRESHOLD_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return default

    models = cfg.get("models") or {}
    entry = models.get(model_name)
    if entry is None:
        return default
    if isinstance(entry, dict):
        return float(entry.get("threshold", default))
    return float(entry)


def load_training_meta() -> dict:
    """读取 train.py 写入的 training_meta.json；缺失或损坏时返回空 dict。"""
    if not TRAINING_META_PATH.is_file():
        return {}
    try:
        with open(TRAINING_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}